"""SFT with LoRA on MBPP train reference solutions, in the same prompt format as GRPO.

Loss is on the completion only. Resumable from the newest checkpoint in
output_dir. Writes log_history.json, timings.json, epoch_<n>/ adapters (one per
pass, for validation-based selection) and TRAINING_DONE. The selected epoch is
merged into a full model by select_and_eval.py for SFT->GRPO to start from.
"""
import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from peft import LoraConfig
from transformers import TrainerCallback, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

from data import load_split
from models import load_model, load_tokenizer
from train_grpo import EpochSaver, exit_on_sigterm

HERE = Path(__file__).resolve().parent


class StepTimer(TrainerCallback):
    def __init__(self, path, rows):
        self.path, self.rows = path, rows

    def on_step_begin(self, args, state, control, **kw):
        self.t0 = time.monotonic()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_step_end(self, args, state, control, **kw):
        row = {"step": state.global_step, "total_s": round(time.monotonic() - self.t0, 2)}
        if torch.cuda.is_available():
            row["peak_gpu_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        self.rows.append(row)
        self.path.write_text(json.dumps(self.rows, indent=1))


def to_sft(row):
    return {"prompt": row["prompt"],
            "completion": [{"role": "assistant", "content": f"```python\n{row['reference'].strip()}\n```"}]}


def main():
    exit_on_sigterm()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    scfg = cfg["sft"]
    out_dir = HERE / scfg["output_dir"]
    scfg["output_dir"] = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "TRAINING_DONE").exists():
        print(f"{out_dir} already finished training; nothing to do")
        return
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg))

    last = get_last_checkpoint(str(out_dir))
    resume_step = int(last.rsplit("-", 1)[1]) if last else 0
    timings = out_dir / "timings.json"
    rows = [r for r in json.loads(timings.read_text()) if r["step"] <= resume_step] if last and timings.exists() else []

    set_seed(cfg["seed"])
    train = load_split("train")
    train = train.map(to_sft, remove_columns=train.column_names)

    trainer = SFTTrainer(
        model=load_model(cfg["dtype"]),
        args=SFTConfig(**scfg),
        train_dataset=train,
        processing_class=load_tokenizer(padding_side="right"),
        peft_config=LoraConfig(task_type="CAUSAL_LM", **cfg["lora"]),
        callbacks=[StepTimer(timings, rows), EpochSaver(out_dir)],
    )
    t0 = time.monotonic()
    trainer.train(resume_from_checkpoint=last)
    wall = time.monotonic() - t0

    (out_dir / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    summary_path = out_dir / "run_summary.json"
    prior = json.loads(summary_path.read_text())["segments"] if summary_path.exists() else []
    segment = {"resumed_from_step": resume_step, "end_step": trainer.state.global_step,
               "wall_seconds": round(wall, 1), "device": str(trainer.model.device),
               "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}
    summary_path.write_text(json.dumps({"segments": prior + [segment]}, indent=1))
    (out_dir / "TRAINING_DONE").write_text(f"{trainer.state.global_step}\n")


if __name__ == "__main__":
    main()
