import json
from pathlib import Path

results = []
for path in sorted(Path("results").glob("*.json")):
    results.append(json.loads(path.read_text()))

print(f"{'version':<8} {'problem':<26} {'verdict':<8} {'steps':<6} {'tokens':<8}")
print("-" * 60)

for r in sorted(results, key=lambda r: (r["version"], r["problem"])):
    tokens = r["input_tokens"] + r["output_tokens"]
    print(f"{r['version']:<8} {r['problem']:<26} {r['verdict']:<8} {r['steps']:<6} {tokens:<8}")

print()
print(f"{'version':<8} {'passed':<10} {'total tokens':<14}")
print("-" * 34)

for version in sorted(set(r["version"] for r in results)):
    rows = [r for r in results if r["version"] == version]
    passed = sum(1 for r in rows if r["verdict"] == "PASS")
    tokens = sum(r["input_tokens"] + r["output_tokens"] for r in rows)
    print(f"{version:<8} {f'{passed}/{len(rows)}':<10} {tokens:<14}")