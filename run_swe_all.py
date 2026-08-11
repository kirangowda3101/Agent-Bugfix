import json
import subprocess
import sys
from pathlib import Path

versions = sys.argv[1:] if len(sys.argv) > 1 else ["swe_v0", "swe_v1", "swe_v2"]

instances = [
    line.strip()
    for line in Path("swe_subset.txt").read_text().splitlines()
    if line.strip()
]

print(f"Instances: {len(instances)}")
print(f"Versions:  {versions}")
print()

for version in versions:
    for instance in instances:
        pred = Path("predictions") / f"{version}_{instance}.jsonl"
        if pred.exists():
            print(f"SKIP {version} {instance} (already done)")
            continue

        print("=" * 60)
        print(f"RUN {version} {instance}")
        print("=" * 60)

        proc = subprocess.run(["python", f"{version}.py", instance])
        if proc.returncode != 0:
            print(f"!!! {version} {instance} exited with {proc.returncode}")

print()
print("All agent runs finished. Combining predictions ...")

for version in versions:
    combined = []
    for instance in instances:
        pred = Path("predictions") / f"{version}_{instance}.jsonl"
        if pred.exists():
            combined.append(pred.read_text().strip())

    if not combined:
        continue

    combined_path = Path("predictions") / f"{version}_all.jsonl"
    combined_path.write_text("\n".join(combined) + "\n")
    print(f"Wrote {combined_path} ({len(combined)} predictions)")

    print(f"Grading {version} ...")
    subprocess.run([
        "python", "-m", "swebench.harness.run_evaluation",
        "--dataset_name", "princeton-nlp/SWE-bench_Verified",
        "--predictions_path", str(combined_path),
        "--instance_ids", *instances,
        "--max_workers", "4",
        "--run_id", f"{version}_full",
        "--cache_level", "env",
    ])