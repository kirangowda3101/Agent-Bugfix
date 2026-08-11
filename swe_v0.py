import json
import os
import sys
from pathlib import Path
from datasets import load_dataset
from dotenv import load_dotenv
from anthropic import Anthropic

instance_id = sys.argv[1]

print("Loading princeton-nlp/SWE-bench_Verified ...")
ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

instance = None
for row in ds:
    if row["instance_id"] == instance_id:
        instance = row
        break

if instance is None:
    print(f"Error: instance {instance_id} not found")
    sys.exit(1)

print(f"Found {instance_id}")

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

prompt = (
    "You are fixing a bug in a Python repository. You cannot see the code.\n\n"
    f"{instance['problem_statement']}\n\n"
    "Produce a unified diff patch that fixes this bug. "
    "Output only the patch in standard git diff format, starting with 'diff --git'. "
    "No explanation, no markdown fences."
)

response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=8000,
    messages=[{"role": "user", "content": prompt}],
)

patch = response.content[0].text.strip()

if patch.startswith("```"):
    lines = patch.splitlines()
    lines = [l for l in lines if not l.strip().startswith("```")]
    patch = "\n".join(lines)

print(patch)

Path("predictions").mkdir(exist_ok=True)
prediction = {
    "instance_id": instance_id,
    "model_name_or_path": "swe_v0",
    "model_patch": patch,
}
(Path("predictions") / f"swe_v0_{instance_id}.jsonl").write_text(json.dumps(prediction) + "\n")

result = {
    "instance_id": instance_id,
    "version": "swe_v0",
    "steps": 1,
    "input_tokens": response.usage.input_tokens,
    "output_tokens": response.usage.output_tokens,
    "patch_length": len(patch),
}

Path("results").mkdir(exist_ok=True)
(Path("results") / f"swe_v0_{instance_id}.json").write_text(json.dumps(result, indent=2))

print()
print("=" * 50)
for key, value in result.items():
    print(f"{key}: {value}")