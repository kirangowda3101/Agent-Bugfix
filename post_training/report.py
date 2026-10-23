"""Build every reported number and plot from the committed files in results/.

Nothing is typed in by hand: results/summary.json, the plots in results/plots/
and the tables printed here (pasted into README.md) all come from
results/*_test.json, results/*_selection.json, results/training/ and
results/audit/.

  python report.py
"""
import json
import math
import statistics
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
PLOTS = RES / "plots"
MODELS = [("base", "Base"), ("sft", "SFT"), ("grpo", "GRPO"), ("sft_grpo", "SFT→GRPO")]
PAIRS = [("base", "sft"), ("base", "grpo"), ("base", "sft_grpo"), ("sft", "sft_grpo"), ("grpo", "sft_grpo"), ("sft", "grpo")]

# Reference palette (dataviz skill), light mode; validated: first three slots, all pairs.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"sft": "#eb6834", "grpo": "#2a78d6", "sft_grpo": "#1baf7a"}


def load(path):
    return json.loads(Path(path).read_text())


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 4), round(c + h, 4)


def mcnemar(a, b):
    """Problems solved by b but not a, by a but not b, and the exact two-sided McNemar p-value."""
    gained = sum(b[t] and not a[t] for t in a)
    lost = sum(a[t] and not b[t] for t in a)
    m = gained + lost
    p = min(1.0, 2 * sum(comb(m, i) for i in range(min(gained, lost) + 1)) / 2 ** m) if m else 1.0
    return gained, lost, round(p, 4)


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def rolling(xs, w=10):
    return [statistics.mean(xs[max(0, i - w + 1):i + 1]) for i in range(len(xs))]


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    summary = {"models": {}, "paired_test_comparisons": [], "training": {}}
    passed = {}

    # --- test results and selection ---------------------------------------------
    for key, label in MODELS:
        test, sel = load(RES / f"{key}_test.json"), load(RES / f"{key}_selection.json")
        rows = test["problems"]
        passed[key] = {r["task_id"]: r["passed_all"] for r in rows}
        k, n = sum(passed[key].values()), len(rows)
        val = {ep: {"pass_at_1": round(v["pass_at_1"], 4), "mean_reward": round(v["mean_reward"], 4)}
               for ep, v in sel["validation"].items()}
        top = max(v["pass_at_1"] for v in val.values())
        summary["models"][key] = {
            "label": label, "test_solved": k, "test_n": n, "test_pass_at_1": round(k / n, 4),
            "test_pass_at_1_wilson95": wilson(k, n), "test_mean_reward": round(test["summary"]["mean_reward"], 4),
            "chosen": sel["chosen"], "validation": val,
            "selection_used_tie_break": sum(v["pass_at_1"] == top for v in val.values()) > 1,
            "gpu": test["config"].get("gpu"), "dtype": test["config"].get("dtype"),
            "sandbox": test["config"].get("sandbox"), "sandbox_restarts": test["summary"]["sandbox_restarts"],
        }
    ids = set(passed["base"])
    assert all(set(p) == ids for p in passed.values()) and len(ids) == 500, "test sets differ"
    for a, b in PAIRS:
        g, lost, p = mcnemar(passed[a], passed[b])
        summary["paired_test_comparisons"].append({"from": a, "to": b, "solved_only_by_to": g, "solved_only_by_from": lost,
                                                   "net": g - lost, "exact_mcnemar_p": p})

    # --- training -----------------------------------------------------------------
    for run in ("sft", "grpo", "sft_grpo"):
        t = load(RES / "training" / run / "timings.json")
        segs = load(RES / "training" / run / "run_summary.json")["segments"]
        entry = {"steps": len(t), "train_wall_seconds": round(sum(s["wall_seconds"] for s in segs), 1),
                 "seconds_per_step_mean": round(statistics.mean(r["total_s"] for r in t), 2),
                 "peak_gpu_gib_max": max(r.get("peak_gpu_gib", 0) for r in t), "gpu": segs[-1].get("gpu")}
        if "generation_s" in t[0]:
            for k in ("generation_s", "scoring_s", "other_s"):
                entry[f"{k}_mean"] = round(statistics.mean(r[k] for r in t), 2)
            entry["sandbox_restarts"] = sum(s.get("sandbox_restarts", 0) for s in segs)
        summary["training"][run] = entry

    audit = load(RES / "audit" / "audit_summary.json")
    reviewed = load(RES / "audit" / "reviewed_hits.json")
    counts = audit["counts"]
    summary["audit"] = {"gate": audit["gate"],
                        "completions_audited": sum(c["completions"] for c in counts.values()),
                        "training_completions_audited": sum(c["completions"] for k, c in counts.items() if k.startswith("rollouts/")),
                        "evaluation_completions_audited": sum(c["completions"] for k, c in counts.items() if not k.startswith("rollouts/")),
                        "reviewed_hits": reviewed["n"],
                        "successful_shortcuts": sum(h["verdict"].startswith("SUCCESSFUL") for h in reviewed["hits"])}
    (RES / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))

    plot_test(summary)
    plot_validation(summary)
    plot_grpo_training()
    plot_sft_loss()
    print_tables(summary)


def plot_test(summary):
    fig, ax = plt.subplots(figsize=(7, 3.0), facecolor=SURFACE)
    style(ax)
    labels = [summary["models"][k]["label"] for k, _ in MODELS][::-1]
    vals = [summary["models"][k] for k, _ in MODELS][::-1]
    y = range(len(vals))
    ax.barh(y, [v["test_pass_at_1"] for v in vals], height=0.5, color="#2a78d6", zorder=2)
    for i, v in enumerate(vals):
        lo, hi = v["test_pass_at_1_wilson95"]
        ax.plot([lo, hi], [i, i], color=INK, linewidth=1.5, zorder=3)
        ax.text(hi + 0.008, i, f"{v['test_pass_at_1']:.3f}  ({v['test_solved']}/{v['test_n']})",
                va="center", fontsize=9, color=INK)
    ax.set_yticks(list(y), labels, color=INK)
    ax.set_xlim(0, 0.55)
    ax.set_xlabel("pass@1 on MBPP test (500 problems); line = 95% Wilson interval", color=INK2, fontsize=9)
    ax.set_title("Test pass@1 by model", color=INK, fontsize=11, loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(PLOTS / "test_pass_at_1.png", dpi=160)
    plt.close(fig)


def plot_validation(summary):
    fig, ax = plt.subplots(figsize=(7, 3.4), facecolor=SURFACE)
    style(ax)
    base = summary["models"]["base"]["validation"]["base"]["pass_at_1"]
    ax.axhline(base, color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(3.05, base - 0.004, f"Base {base:.3f}", va="top", fontsize=9, color=INK2)
    for key in ("sft", "grpo", "sft_grpo"):
        m = summary["models"][key]
        eps = sorted(m["validation"], key=lambda e: int(e.split("_")[1]))
        xs, ys = [int(e.split("_")[1]) for e in eps], [m["validation"][e]["pass_at_1"] for e in eps]
        ax.plot(xs, ys, color=SERIES[key], linewidth=2, marker="o", markersize=6, label=m["label"], zorder=3)
        c = int(m["chosen"].split("_")[1])
        ax.plot([c], [m["validation"][m["chosen"]]["pass_at_1"]], marker="o", markersize=13, markerfacecolor="none",
                markeredgecolor=SERIES[key], markeredgewidth=1.5, zorder=4)
        ax.text(3.05, ys[-1], m["label"], va="center", fontsize=9, color=INK)
    ax.set_xticks([1, 2, 3], ["epoch 1", "epoch 2", "epoch 3"])
    ax.set_xlim(0.8, 3.6)
    ax.set_ylabel("validation pass@1 (90 problems)", color=INK2, fontsize=9)
    ax.set_title("Validation pass@1 per epoch (ring = epoch chosen for the test run)", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(PLOTS / "validation_by_epoch.png", dpi=160)
    plt.close(fig)


def plot_grpo_training():
    panels = [("reward", "mean reward per step"), ("kl", "KL to reference"),
              ("completions/mean_length", "mean completion length (tokens)")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.5, 7.2), sharex=True, facecolor=SURFACE)
    for run, label in (("grpo", "GRPO"), ("sft_grpo", "SFT→GRPO")):
        h = [r for r in load(RES / "training" / run / "log_history.json") if "reward" in r]
        steps = [r["step"] for r in h]
        for ax, (key, _) in zip(axes, panels):
            ys = [r[key] for r in h]
            ax.plot(steps, ys, color=SERIES[run], linewidth=0.8, alpha=0.3, zorder=2)
            ax.plot(steps, rolling(ys), color=SERIES[run], linewidth=2, label=f"{label} (10-step mean)", zorder=3)
    for ax, (_, title) in zip(axes, panels):
        style(ax)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        for b in (46, 92):
            ax.axvline(b + 0.5, color=INK2, linewidth=0.8, linestyle=(0, (2, 3)), zorder=1)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    axes[0].text(47, axes[0].get_ylim()[1], " epoch 2", va="top", fontsize=8, color=INK2)
    axes[0].text(93, axes[0].get_ylim()[1], " epoch 3", va="top", fontsize=8, color=INK2)
    axes[-1].set_xlabel("optimizer step (64 completions each; thin line = per step)", color=INK2, fontsize=9)
    fig.suptitle("GRPO training on MBPP train", color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(PLOTS / "grpo_training.png", dpi=160)
    plt.close(fig)


def plot_sft_loss():
    h = [r for r in load(RES / "training" / "sft" / "log_history.json") if "loss" in r and "step" in r]
    fig, ax = plt.subplots(figsize=(7, 2.8), facecolor=SURFACE)
    style(ax)
    steps, ys = [r["step"] for r in h], [r["loss"] for r in h]
    ax.plot(steps, ys, color=SERIES["sft"], linewidth=0.8, alpha=0.3)
    ax.plot(steps, rolling(ys), color=SERIES["sft"], linewidth=2)
    for b in (47, 94):
        ax.axvline(b + 0.5, color=INK2, linewidth=0.8, linestyle=(0, (2, 3)))
    ax.set_xlabel("optimizer step (8 examples each)", color=INK2, fontsize=9)
    ax.set_title("SFT training loss (10-step mean; thin line = per step)", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(PLOTS / "sft_loss.png", dpi=160)
    plt.close(fig)


def print_tables(s):
    print("| Model | Epoch chosen on validation | Test pass@1 | 95% interval | Solved | Test mean reward |")
    print("|---|---|---|---|---|---|")
    for key, label in MODELS:
        m = s["models"][key]
        lo, hi = m["test_pass_at_1_wilson95"]
        print(f"| {label} | {m['chosen'].replace('_', ' ')}{' (tie-break)' if m['selection_used_tie_break'] else ''} "
              f"| {m['test_pass_at_1']:.3f} | {lo:.3f}–{hi:.3f} | {m['test_solved']}/{m['test_n']} | {m['test_mean_reward']:.3f} |")
    print()
    print("| Comparison | Solved only by the second | Solved only by the first | Net | Exact McNemar p |")
    print("|---|---|---|---|---|")
    lab = dict(MODELS)
    for c in s["paired_test_comparisons"]:
        print(f"| {lab[c['from']]} → {lab[c['to']]} | {c['solved_only_by_to']} | {c['solved_only_by_from']} "
              f"| {c['net']:+d} | {c['exact_mcnemar_p']:.3f} |")
    print()
    print("| Run | Epoch | Validation pass@1 | Validation mean reward |")
    print("|---|---|---|---|")
    for key, label in MODELS:
        for ep, v in s["models"][key]["validation"].items():
            print(f"| {label} | {ep.replace('_', ' ')} | {v['pass_at_1']:.3f} | {v['mean_reward']:.3f} |")
    print()
    print(json.dumps({"training": s["training"], "audit": s["audit"]}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
