"""Pick a run's checkpoint on VALIDATION, then evaluate only that one on TEST.

Selection rule (fixed before any results): highest validation pass@1; ties go
to higher validation mean reward, then to the earlier epoch.

Every evaluation is cached, so rerunning after a timeout skips finished work.

  python select_and_eval.py --name base                       # base model, no selection
  python select_and_eval.py --name sft --run runs/sft --merge-to runs/sft/best_merged
  python select_and_eval.py --name grpo --run runs/grpo
  python select_and_eval.py --name sft_grpo --run runs/sft_grpo --weights runs/sft/best_merged
"""
import argparse
import json
from pathlib import Path

from transformers import set_seed

from evaluate import evaluate
from models import device, gpu_name, load_model, load_tokenizer
from sandbox import make_sandbox

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def cached_eval(path, tok, sb, args, weights, adapter, split):
    if path.exists():
        return json.loads(path.read_text())["summary"]
    model = load_model(args.dtype, weights, adapter).to(device()).eval()
    summary, rows = evaluate(model, tok, split, sb, limit=args.limit, batch=args.batch)
    cfg = {"weights": weights, "adapter": adapter, "split": split, "dtype": args.dtype,
           "batch": args.batch, "seed": args.seed, "device": device(), "gpu": gpu_name(), "sandbox": sb.backend}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"config": cfg, "summary": summary, "problems": rows}, indent=1))
    del model
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--run", default=None, help="training output dir containing epoch_<n>/ adapters")
    ap.add_argument("--weights", default=None, help="base weights the adapters were trained on (SFT->GRPO)")
    ap.add_argument("--merge-to", default=None, help="save the selected adapter merged into full weights here")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results-dir", default=str(RESULTS))
    ap.add_argument("--limit", type=int, default=None, help="dry runs only: first N problems")
    ap.add_argument("--skip-test", action="store_true", help="dry runs only: never touch the test split")
    args = ap.parse_args()
    results = Path(args.results_dir)
    assert args.limit is None or (args.skip_test and results != RESULTS), \
        "--limit is for dry runs: use it with --skip-test and a separate --results-dir"

    set_seed(args.seed)
    weights = str(HERE / args.weights) if args.weights else None
    tok = load_tokenizer()
    with make_sandbox(timeout=5.0, workers=4) as sb:
        assert sb.seccomp_active(), "seccomp filter not applied to sandbox"

        if args.run is None:
            candidates = {"base": None}
        else:
            run = HERE / args.run
            assert (run / "TRAINING_DONE").exists(), f"{run} has not finished training"
            epochs = sorted(run.glob("epoch_*"), key=lambda p: int(p.name.split("_")[1]))
            assert epochs, f"no epoch_<n> adapters in {run}"
            candidates = {p.name: str(p) for p in epochs}

        val = {}
        for name, adapter in candidates.items():
            val[name] = cached_eval(results / "validation" / f"{args.name}_{name}.json",
                                    tok, sb, args, weights, adapter, "validation")
            print(f"validation {args.name} {name}: {json.dumps(val[name])}", flush=True)

        order = list(candidates)
        chosen = max(order, key=lambda n: (val[n]["pass_at_1"], val[n]["mean_reward"], -order.index(n)))
        selection = {"name": args.name, "chosen": chosen, "adapter": candidates[chosen], "weights": weights,
                     "rule": "max validation pass@1, then mean reward, then earliest epoch",
                     "validation": val}
        (results / f"{args.name}_selection.json").write_text(json.dumps(selection, indent=1))
        print(f"selected {args.name}: {chosen}", flush=True)

        if not args.skip_test:
            test = cached_eval(results / f"{args.name}_test.json", tok, sb, args, weights, candidates[chosen], "test")
            print(f"TEST {args.name} ({chosen}): {json.dumps(test)}", flush=True)

    if args.merge_to:
        out = HERE / args.merge_to
        if not (out / "config.json").exists():
            load_model(args.dtype, weights, candidates[chosen]).save_pretrained(out)
            tok.save_pretrained(out)
        print(f"merged {chosen} into {out}")


if __name__ == "__main__":
    main()
