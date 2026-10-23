"""Greedy pass@1 on an MBPP split for the base model, a merged SFT model, and/or a LoRA adapter.

pass@1 = fraction of problems where the single greedy completion passes ALL
asserts. The test split is only for final, already-selected models; use
validation for everything else.
"""
import argparse
import json
import time
from pathlib import Path

from transformers import set_seed

from check_reward_signal import generate
from data import load_split
from models import device, gpu_name, load_model, load_tokenizer
from rewards import score
from sandbox import make_sandbox


def evaluate(model, tok, split, sandbox, limit=None, batch=8, max_new_tokens=512):
    ds = load_split(split)
    if limit:
        ds = ds.select(range(limit))
    rows, t_gen, t_score = [], 0.0, 0.0
    for b in range(0, len(ds), batch):
        chunk = ds.select(range(b, min(b + batch, len(ds))))
        t0 = time.monotonic()
        comps = generate(model, tok, chunk["prompt"], 1, 0.0, max_new_tokens)
        t1 = time.monotonic()
        rewards, details = score(sandbox, comps, chunk["test_list"], chunk["test_setup_code"])
        t_score += time.monotonic() - t1
        t_gen += t1 - t0
        for p, c, r, d in zip(chunk, comps, rewards, details):
            rows.append({"task_id": p["task_id"], "reward": r, "passed_all": r == 1.0,
                         "completion": c, "errors": d["errors"]})
        print(f"{split}: {len(rows)}/{len(ds)}  gen {t_gen:.0f}s  score {t_score:.0f}s", flush=True)
    summary = {
        "split": split,
        "n_problems": len(rows),
        "pass_at_1": sum(r["passed_all"] for r in rows) / len(rows),
        "mean_reward": sum(r["reward"] for r in rows) / len(rows),
        "seconds_generation": round(t_gen, 1),
        "seconds_scoring": round(t_score, 1),
        "sandbox_restarts": sandbox.restarts,
    }
    return summary, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="local merged model (e.g. best SFT); default: pinned base")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    tok = load_tokenizer()
    model = load_model(args.dtype, args.weights, args.adapter).to(device()).eval()
    with make_sandbox(timeout=5.0, workers=4) as sb:
        assert sb.seccomp_active(), "seccomp filter not applied to sandbox"
        summary, rows = evaluate(model, tok, args.split, sb, args.limit, args.batch, args.max_new_tokens)
    out = {"config": vars(args) | {"device": device(), "gpu": gpu_name(), "sandbox": sb.backend}, "summary": summary, "problems": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
