"""Measure whether the base model gives GRPO a usable reward signal on MBPP train.

GRPO's advantage is reward minus the group mean, so a problem only produces a
gradient when the sampled completions for it get different rewards. This script
samples a fixed, seeded subset of TRAIN problems (never test) and reports:
  - greedy pass@1 (all asserts pass) and mean greedy reward
  - sampled pass@1 and mean sampled reward at the given temperature
  - fraction of problems whose sampled rewards are not all equal ("mixed"),
    i.e. the problems GRPO can actually learn from
"""
import argparse
import json
import platform
import random
import statistics
import time
from pathlib import Path

import torch
from transformers import set_seed

from data import MODEL, MODEL_REVISION, load_split
from models import device as pick_device, load_model, load_tokenizer
from rewards import score
from sandbox import make_sandbox

RESULTS = Path(__file__).resolve().parent / "results"


def generate(model, tok, prompts, n, temperature, max_new_tokens):
    texts = [tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    kwargs = dict(max_new_tokens=max_new_tokens, num_return_sequences=n, pad_token_id=tok.pad_token_id)
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature, top_p=1.0, top_k=0)
    else:
        kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    with torch.no_grad():
        out = model.generate(**enc, **kwargs)
    gen = out[:, enc["input_ids"].shape[1]:]
    return tok.batch_decode(gen, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-problems", type=int, default=50)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-prompts", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(RESULTS / "reward_signal_base.json"))
    args = ap.parse_args()

    set_seed(args.seed)
    device = pick_device()
    tok = load_tokenizer()
    model = load_model("float32").to(device).eval()

    train = load_split("train")
    idx = sorted(random.Random(args.seed).sample(range(len(train)), args.n_problems))
    probs = [train[i] for i in idx]

    problems = []
    t_gen = t_score = 0.0
    with make_sandbox(timeout=5.0, workers=4) as sb:
        assert sb.seccomp_active(), "seccomp filter not applied to sandbox"
        for b in range(0, len(probs), args.batch_prompts):
            batch = probs[b:b + args.batch_prompts]
            prompts = [p["prompt"] for p in batch]
            t0 = time.monotonic()
            greedy = generate(model, tok, prompts, 1, 0.0, args.max_new_tokens)
            sampled = generate(model, tok, prompts, args.samples, args.temperature, args.max_new_tokens)
            t1 = time.monotonic()
            # sampled is grouped per prompt: [p0 s0..s7, p1 s0..s7, ...]
            comps = greedy + sampled
            tests = [p["test_list"] for p in batch] + [p["test_list"] for p in batch for _ in range(args.samples)]
            setups = [p["test_setup_code"] for p in batch] + [p["test_setup_code"] for p in batch for _ in range(args.samples)]
            rewards, details = score(sb, comps, tests, setups)
            t_score += time.monotonic() - t1
            t_gen += t1 - t0
            for k, p in enumerate(batch):
                s0 = len(batch) + k * args.samples
                problems.append({
                    "task_id": p["task_id"],
                    "greedy": {"reward": rewards[k], "completion": comps[k], "errors": details[k]["errors"]},
                    "samples": [{"reward": rewards[j], "completion": comps[j], "errors": details[j]["errors"]}
                                for j in range(s0, s0 + args.samples)],
                })
            print(f"{len(problems)}/{len(probs)} problems  gen {t_gen:.0f}s  score {t_score:.0f}s", flush=True)
        restarts = sb.restarts

    def all_pass(r):
        return r == 1.0

    sample_rewards = [[s["reward"] for s in p["samples"]] for p in problems]
    all_comps = [p["greedy"] for p in problems] + [s for p in problems for s in p["samples"]]
    summary = {
        "n_problems": len(problems),
        "greedy_pass_at_1": sum(all_pass(p["greedy"]["reward"]) for p in problems) / len(problems),
        "greedy_mean_reward": statistics.mean(p["greedy"]["reward"] for p in problems),
        "sampled_pass_at_1": statistics.mean(all_pass(r) for rs in sample_rewards for r in rs),
        "sampled_mean_reward": statistics.mean(r for rs in sample_rewards for r in rs),
        "frac_problems_mixed_reward": sum(len(set(rs)) > 1 for rs in sample_rewards) / len(problems),
        "frac_problems_all_zero": sum(all(r == 0 for r in rs) for rs in sample_rewards) / len(problems),
        "frac_problems_all_one": sum(all(r == 1 for r in rs) for rs in sample_rewards) / len(problems),
        "frac_problems_any_sample_passes": sum(any(all_pass(r) for r in rs) for rs in sample_rewards) / len(problems),
        "frac_completions_no_code_block": sum(c["errors"][:1] == ["no code block"] for c in all_comps) / len(all_comps),
        "frac_completions_timeout": sum("timeout" in c["errors"] for c in all_comps) / len(all_comps),
        "sandbox_restarts": restarts,
        "seconds_generation": round(t_gen, 1),
        "seconds_scoring": round(t_score, 1),
    }
    out = {
        "config": vars(args) | {"model": MODEL, "model_revision": MODEL_REVISION, "device": device, "dtype": "float32", "split": "train",
                                "task_ids": [p["task_id"] for p in problems],
                                "torch": torch.__version__, "platform": platform.platform()},
        "summary": summary,
        "problems": problems,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
