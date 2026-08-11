import base64
import json
import os
import subprocess
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

fail_to_pass = json.loads(instance["FAIL_TO_PASS"])
pass_to_pass = json.loads(instance["PASS_TO_PASS"])
print(f"Found {instance_id}: {len(fail_to_pass)} FAIL_TO_PASS, {len(pass_to_pass)} PASS_TO_PASS")

repo_part, name_part = instance_id.split("__", 1)
image = f"swebench/sweb.eval.x86_64.{repo_part}_1776_{name_part}:latest"
container_name = f"swe_agent_{instance_id}"
print(f"Image: {image}")


def run_in_container(cmd):
    proc = subprocess.run(
        ["docker", "exec", "-w", "/testbed", container_name, "bash", "-c", cmd],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return proc.stdout + proc.stderr


def search_code(pattern):
    output = run_in_container(f"grep -rn --include=*.py '{pattern}' .")
    lines = output.splitlines()
    if len(lines) > 30:
        return "\n".join(lines[:30]) + f"\n... ({len(lines) - 30} more matches)"
    return output


def read_file(filename):
    output = run_in_container(f"cat {filename}")
    lines = output.splitlines()
    if len(lines) > 400:
        return "\n".join(lines[:400]) + f"\n... (truncated, {len(lines) - 400} more lines)"
    return output


def write_file(filename, content):
    encoded = base64.b64encode(content.encode()).decode()
    run_in_container(f"echo {encoded} | base64 -d > /testbed/{filename}")
    return f"Wrote {len(content)} characters to {filename}"


print(f"Removing any existing container named {container_name} ...")
subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

print(f"Starting container {container_name} ...")
subprocess.run(
    ["docker", "run", "-d", "--name", container_name, image, "sleep", "3600"],
    check=True,
)

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

tools = [
    {
        "name": "search_code",
        "description": "Search the repository for a pattern using grep. Returns matching file:line:content results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The grep pattern to search for, e.g. def get_queryset",
                }
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file in the repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Path to the file, relative to /testbed, e.g. django/db/models/query.py",
                }
            },
            "required": ["filename"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file in the repository. Overwrites the file completely.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Path to the file, relative to /testbed, e.g. django/db/models/query.py",
                },
                "content": {
                    "type": "string",
                    "description": "The complete new contents of the file",
                },
            },
            "required": ["filename", "content"],
        },
    },
]

prompt = (
    "You are working in a Django repository checked out at /testbed inside a Docker container.\n\n"
    f"{instance['problem_statement']}\n\n"
    "Locate the bug described above and fix it using the available tools."
)

messages = [{"role": "user", "content": prompt}]

MAX_STEPS = 20
step = 0
input_tokens = 0
output_tokens = 0

try:
    while step < MAX_STEPS:
        step += 1
        print(f"--- step {step} ---")

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=8000,
            tools=tools,
            messages=messages,
        )

        input_tokens += response.usage.input_tokens
        output_tokens += response.usage.output_tokens

        for block in response.content:
            if block.type == "text":
                print("CLAUDE:", block.text)

        if response.stop_reason != "tool_use":
            break

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print("TOOL CALL:", block.name, block.input)

                if block.name == "search_code":
                    result = search_code(block.input["pattern"])

                elif block.name == "read_file":
                    result = read_file(block.input["filename"])

                elif block.name == "write_file":
                    result = write_file(block.input["filename"], block.input["content"])

                else:
                    result = f"Error: unknown tool {block.name}"

                print("TOOL RESULT:", result)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        messages.append({"role": "user", "content": tool_results})

    print("Collecting patch ...")
    patch = run_in_container("git -C /testbed diff")
finally:
    print(f"Stopping and removing container {container_name} ...")
    subprocess.run(["docker", "stop", container_name], capture_output=True)
    subprocess.run(["docker", "rm", container_name], capture_output=True)

Path("predictions").mkdir(exist_ok=True)
prediction = {
    "instance_id": instance_id,
    "model_name_or_path": "swe_v1",
    "model_patch": patch,
}
(Path("predictions") / f"swe_v1_{instance_id}.jsonl").write_text(json.dumps(prediction) + "\n")

result = {
    "instance_id": instance_id,
    "version": "swe_v1",
    "steps": step,
    "input_tokens": input_tokens,
    "output_tokens": output_tokens,
    "patch_length": len(patch),
}

Path("results").mkdir(exist_ok=True)
(Path("results") / f"swe_v1_{instance_id}.json").write_text(json.dumps(result, indent=2))

print()
print("=" * 50)
for key, value in result.items():
    print(f"{key}: {value}")