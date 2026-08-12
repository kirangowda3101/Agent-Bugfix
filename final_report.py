import json
from pathlib import Path

RESULTS = Path("results")


def load(pattern, exclude=None):
    files = sorted(RESULTS.glob(pattern))
    if exclude:
        files = [f for f in files if not any(x in f.name for x in exclude)]
    return [json.loads(f.read_text()) for f in files]


def totals(rows):
    tok = sum(r["input_tokens"] + r["output_tokens"] for r in rows)
    steps = sum(r["steps"] for r in rows)
    return len(rows), tok, steps


def graded(version):
    path = Path(f"{version}.{version}_full.json")
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return {
        "resolved": d["resolved_instances"],
        "total": d["total_instances"],
        "errors": d["error_instances"],
        "empty": d["empty_patch_instances"],
        "ids": sorted(d["resolved_ids"]),
    }


V2_EXCLUDE = ["_rag_", "_mcp_"]

print("=" * 74)
print("DIMENSION 1 - ENGINEERING LAYERS (curated problems)")
print("=" * 74)
print()

curated = load("v*.json")
if curated:
    by_version = {}
    for r in curated:
        by_version.setdefault(r["version"], []).append(r)

    print(f"{'version':<10} {'passed':<10} {'tokens':<12} {'steps':<8}")
    print("-" * 44)
    for v in sorted(by_version):
        rows = by_version[v]
        passed = sum(1 for r in rows if r["verdict"] == "PASS")
        n, tok, steps = totals(rows)
        score = f"{passed}/{n}"
        print(f"{v:<10} {score:<10} {tok:<12,} {steps:<8}")
else:
    print("(no curated results found)")

print()
print("=" * 74)
print("DIMENSION 1 - ENGINEERING LAYERS (SWE-bench Verified)")
print("=" * 74)
print()

swe_versions = [
    ("swe_v0", "one-shot, no tools"),
    ("swe_v1", "tools, cannot run tests"),
    ("swe_v2", "tools + test execution"),
]

print(f"{'version':<12} {'resolved':<10} {'tokens':<12} {'steps':<8} notes")
print("-" * 74)
for v, note in swe_versions:
    g = graded(v)
    if g is None:
        print(f"{v:<12} (not graded)")
        continue
    rows = load(f"{v}_*.json", exclude=V2_EXCLUDE if v == "swe_v2" else None)
    n, tok, steps = totals(rows)
    score = f"{g['resolved']}/{g['total']}"
    print(f"{v:<12} {score:<10} {tok:<12,} {steps:<8} {note}")

print()
print("=" * 74)
print("RETRIEVAL - GREP vs SEMANTIC (SWE-bench Verified)")
print("=" * 74)
print()

print(f"{'method':<14} {'resolved':<10} {'tokens':<12} index build")
print("-" * 56)

g_grep = graded("swe_v2")
rows_grep = load("swe_v2_*.json", exclude=V2_EXCLUDE)
_, tok_grep, _ = totals(rows_grep)
score_grep = f"{g_grep['resolved']}/{g_grep['total']}"
print(f"{'grep':<14} {score_grep:<10} {tok_grep:<12,} 0s")

g_rag = graded("swe_v2_rag")
rows_rag = load("swe_v2_rag_*.json")
if g_rag:
    _, tok_rag, _ = totals(rows_rag)
    idx = sum(r.get("index_build_seconds", 0) for r in rows_rag)
    score_rag = f"{g_rag['resolved']}/{g_rag['total']}"
    print(f"{'semantic':<14} {score_rag:<10} {tok_rag:<12,} {idx:.0f}s total")
    print()
    print("grep solved:    ", g_grep["ids"])
    print("semantic solved:", g_rag["ids"])

print()
print("=" * 74)
print("DIMENSION 2 - SINGLE vs MULTI-AGENT (SWE-bench Verified)")
print("=" * 74)
print()

print(f"{'architecture':<16} {'resolved':<10} {'tokens':<12}")
print("-" * 42)

g_single = graded("swe_v2")
rows_single = load("swe_v2_*.json", exclude=V2_EXCLUDE)
_, tok_single, _ = totals(rows_single)
score_single = f"{g_single['resolved']}/{g_single['total']}"
print(f"{'single agent':<16} {score_single:<10} {tok_single:<12,}")

g_multi = graded("swe_multi")
rows_multi = load("swe_multi_*.json")
if g_multi:
    _, tok_multi, _ = totals(rows_multi)
    score_multi = f"{g_multi['resolved']}/{g_multi['total']}"
    print(f"{'multi-agent':<16} {score_multi:<10} {tok_multi:<12,}")
    delta = (tok_multi - tok_single) / tok_single * 100
    print()
    print(f"token difference: {delta:+.0f}%")
    print()
    print("single solved:", g_single["ids"])
    print("multi solved: ", g_multi["ids"])

print()
print("=" * 74)
print("MCP REFACTOR CHECK")
print("=" * 74)
print()

rows_mcp = load("swe_v2_mcp_*.json")
if rows_mcp:
    for r in rows_mcp:
        tok = r["input_tokens"] + r["output_tokens"]
        print(f"{r['instance_id']}: {tok:,} tokens, patch {r['patch_length']} chars")
    print()
    print("(compare patch_length against swe_v2 for the same instance)")
else:
    print("(no mcp results found)")