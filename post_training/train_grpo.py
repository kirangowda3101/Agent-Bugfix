"""GRPO with LoRA on MBPP train. Reward = fraction of asserts passed in the sandbox.

Resumable: rerunning the same config continues from the newest checkpoint in
output_dir (optimizer, scheduler, RNG and data position included), so a job
that hits the SLURM time limit loses at most `save_steps` steps.

Writes to <output_dir>/:
  log_history.json  TRL's per-step metrics (reward, reward_std, kl, lengths, ...)
  timings.json      per-step wall time split into generation / scoring / rest, peak GPU memory
  rollouts.jsonl    every scored completion, for the reward-hacking audit
  epoch_<n>/        LoRA adapter at the end of each pass over the data (for selection)
  checkpoint-<k>/   rolling checkpoints for resuming
  TRAINING_DONE     written when training finished
"""
import argparse
import json
import math
import signal
import sys
import time
from pathlib import Path

import torch
import yaml
from peft import LoraConfig
from transformers import TrainerCallback, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer

from data import load_split
from models import load_model, load_tokenizer
from rewards import make_reward_fn
from sandbox import make_sandbox

HERE = Path(__file__).resolve().parent


class TimedGRPOTrainer(GRPOTrainer):
    """Records how long generation takes inside each optimizer step."""
    gen_seconds = 0.0

    def _generate(self, prompts):
        t0 = time.monotonic()
        out = super()._generate(prompts)
        TimedGRPOTrainer.gen_seconds += time.monotonic() - t0
        return out


class StepTimer(TrainerCallback):
    def __init__(self, path, score_seconds, rows):
        self.path, self.score_seconds, self.rows = path, score_seconds, rows

    def on_step_begin(self, args, state, control, **kw):
        self.t0 = time.monotonic()
        TimedGRPOTrainer.gen_seconds = 0.0
        self.score_seconds[0] = 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_step_end(self, args, state, control, **kw):
        total = time.monotonic() - self.t0
        gen, score = TimedGRPOTrainer.gen_seconds, self.score_seconds[0]
        row = {"step": state.global_step, "total_s": round(total, 2), "generation_s": round(gen, 2),
               "scoring_s": round(score, 2), "other_s": round(total - gen - score, 2)}
        if torch.cuda.is_available():
            row["peak_gpu_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        self.rows.append(row)
        print(json.dumps(row), flush=True)
        self.path.write_text(json.dumps(self.rows, indent=1))


class EpochSaver(TrainerCallback):
    def __init__(self, out_dir):
        self.out_dir = out_dir

    def on_epoch_end(self, args, state, control, model=None, **kw):
        n = math.ceil(state.epoch - 1e-6)  # a run cut short by max_steps saves its partial pass as epoch_1
        model.save_pretrained(self.out_dir / f"epoch_{n}")
        print(f"saved epoch_{n} at step {state.global_step}", flush=True)


def trim_logs(out_dir, resume_step):
    """Drop log rows from steps after the checkpoint being resumed; they will be redone.

    Rollouts are logged with the global_step at generation time, i.e. step k's
    completions carry k-1, so rows with step < resume_step belong to saved steps.
    """
    rollouts = out_dir / "rollouts.jsonl"
    if rollouts.exists():
        keep = [l for l in rollouts.read_text().splitlines() if json.loads(l)["step"] < resume_step]
        rollouts.write_text("".join(l + "\n" for l in keep))
    timings = out_dir / "timings.json"
    rows = [r for r in json.loads(timings.read_text()) if r["step"] <= resume_step] if timings.exists() else []
    return rows


def exit_on_sigterm():
    """SLURM sends SIGTERM at the time limit; exit normally so the sandbox is cleaned up."""
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))


def main():
    exit_on_sigterm()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    gcfg = cfg["grpo"]
    out_dir = HERE / gcfg["output_dir"]
    gcfg["output_dir"] = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "TRAINING_DONE").exists():
        print(f"{out_dir} already finished training; nothing to do")
        return
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg))

    last = get_last_checkpoint(str(out_dir))
    resume_step = int(last.rsplit("-", 1)[1]) if last else 0
    timing_rows = trim_logs(out_dir, resume_step) if last else []
    if last:
        print(f"resuming from {last}", flush=True)
    else:
        (out_dir / "rollouts.jsonl").write_text("")

    set_seed(cfg["seed"])
    train = load_split("train")
    if cfg.get("max_train_problems"):
        train = train.select(range(cfg["max_train_problems"]))

    tok = load_tokenizer()
    model = load_model(cfg["dtype"], weights=cfg.get("init_from") and str(HERE / cfg["init_from"]))

    rollouts = open(out_dir / "rollouts.jsonl", "a")
    score_seconds = [0.0]

    def log(completions, rewards, details, kwargs):
        step = kwargs["trainer_state"].global_step
        for c, r, d, tid in zip(completions, rewards, details, kwargs["task_id"]):
            text = c if isinstance(c, str) else c[-1]["content"]
            rollouts.write(json.dumps({"step": step, "task_id": tid, "reward": r,
                                       "completion": text, "errors": d["errors"]}) + "\n")
        rollouts.flush()

    with make_sandbox(**cfg["sandbox"]) as sb:
        assert sb.seccomp_active(), "seccomp filter not applied to sandbox"
        inner = make_reward_fn(sb, log=log)

        def mbpp_pass_fraction(**kw):
            t0 = time.monotonic()
            r = inner(**kw)
            score_seconds[0] += time.monotonic() - t0
            return r

        trainer = TimedGRPOTrainer(
            model=model,
            reward_funcs=mbpp_pass_fraction,
            args=GRPOConfig(**gcfg),
            train_dataset=train,
            processing_class=tok,
            peft_config=LoraConfig(task_type="CAUSAL_LM", **cfg["lora"]),
            callbacks=[StepTimer(out_dir / "timings.json", score_seconds, timing_rows), EpochSaver(out_dir)],
        )
        t0 = time.monotonic()
        trainer.train(resume_from_checkpoint=last)
        wall = time.monotonic() - t0
        restarts = sb.restarts

    rollouts.close()
    (out_dir / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    summary_path = out_dir / "run_summary.json"
    prior = json.loads(summary_path.read_text())["segments"] if summary_path.exists() else []
    segment = {"resumed_from_step": resume_step, "end_step": trainer.state.global_step,
               "wall_seconds": round(wall, 1), "sandbox_restarts": restarts, "sandbox": sb.backend,
               "device": str(trainer.model.device),
               "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}
    summary_path.write_text(json.dumps({"segments": prior + [segment]}, indent=1))
    if cfg.get("save_adapter", True):
        trainer.model.save_pretrained(out_dir / "adapter")
    (out_dir / "TRAINING_DONE").write_text(f"{trainer.state.global_step}\n")


if __name__ == "__main__":
    main()
