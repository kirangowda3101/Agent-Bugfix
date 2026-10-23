"""Collect the GPU smoke-test measurements into one JSON file and print it.

Only measured values go in. Nothing here estimates the full run; that is done
afterwards, from these numbers, and labelled as an extrapolation.
"""
import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from sandbox import make_sandbox  # noqa: E402


def stats(values):
    return {"mean": round(statistics.mean(values), 2), "min": min(values), "max": max(values)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    run = HERE / "runs" / "grpo_gpu_smoke"
    timings = json.loads((run / "timings.json").read_text())
    history = [h for h in json.loads((run / "log_history.json").read_text()) if "reward" in h]
    eval_ = json.loads((HERE / "runs" / "eval_gpu_smoke_val20.json").read_text())["summary"]

    # Fixed cost of one sandbox batch: a fresh `apptainer exec` per call.
    with make_sandbox(timeout=5.0, workers=4) as sb:
        job = [{"id": "0", "code": "x = 1", "setup": "", "tests": ["assert x == 1"]}]
        overhead = []
        for _ in range(5):
            t0 = time.monotonic()
            assert sb.run(job)[0]["passed"] == [True]
            overhead.append(round(time.monotonic() - t0, 2))

    summary = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "apptainer": subprocess.run(["apptainer", "--version"], capture_output=True, text=True).stdout.strip(),
        "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                 text=True, cwd=HERE).stdout.strip(),
        "grpo_steps": len(timings),
        "grpo_seconds_per_step": {k: stats([r[k] for r in timings])
                                  for k in ("total_s", "generation_s", "scoring_s", "other_s", "peak_gpu_gib")},
        "grpo_per_step": timings,
        "grpo_metrics": [{k: h.get(k) for k in ("step", "reward", "reward_std", "frac_reward_zero_std", "kl",
                                                "completions/mean_length", "completions/clipped_ratio")}
                         for h in history],
        "grpo_run": json.loads((run / "run_summary.json").read_text()),
        "checkpoint_saved": (run / "checkpoint-5").is_dir(),
        "eval_val20": eval_,
        "sandbox_batch_overhead_s": overhead,
    }
    Path(args.out).write_text(json.dumps(summary, indent=1))
    print("SMOKE SUMMARY")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
