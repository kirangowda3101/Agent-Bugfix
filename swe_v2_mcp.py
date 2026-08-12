import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from datasets import load_dataset
from dotenv import load_dotenv
from anthropic import Anthropic
from mcp import Client
import mcp_server

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

print(f"Removing any existing container named {container_name} ...")
subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

print(f"Starting container {container_name} ...")
subprocess.run(
    ["docker", "run", "-d", "--name", container_name, image, "sleep", "3600"],
    check=True,
)

mcp_server.set_container(container_name)

load_dotenv()
anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

prompt = (
    "You are working in a Django repository checked out at /testbed inside a Docker container.\n\n"
    f"{instance['problem_statement']}\n\n"
    "Locate the bug described above and fix it using the available tools. "
    "When you believe the fix is complete, run the relevant tests to confirm."
)

MAX_STEPS = 20


async def main():
    messages = [{"role": "user", "content": prompt}]
    step = 0
    input_tokens = 0
    output_tokens = 0

    try:
        async with Client(mcp_server.mcp) as client:
            tools_result = await client.list_tools()
            tools = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in tools_result.tools
            ]
            print("Tools from MCP server:", [t["name"] for t in tools])

            while step < MAX_STEPS:
                step += 1
                print(f"--- step {step} ---")

                response = anthropic.messages.create(
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

                        r = await client.call_tool(block.name, block.input)
                        result = r.content[0].text

                        print("TOOL RESULT:", result)

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                messages.append({"role": "user", "content": tool_results})

        print("Collecting patch ...")
        patch = mcp_server.run_in_container("git -C /testbed diff")
    finally:
        print(f"Stopping and removing container {container_name} ...")
        subprocess.run(["docker", "stop", container_name], capture_output=True)
        subprocess.run(["docker", "rm", container_name], capture_output=True)

    Path("predictions").mkdir(exist_ok=True)
    prediction = {
        "instance_id": instance_id,
        "model_name_or_path": "swe_v2_mcp",
        "model_patch": patch,
    }
    (Path("predictions") / f"swe_v2_mcp_{instance_id}.jsonl").write_text(
        json.dumps(prediction) + "\n"
    )

    result = {
        "version": "swe_v2_mcp",
        "instance_id": instance_id,
        "steps": step,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "patch_length": len(patch),
    }

    Path("results").mkdir(exist_ok=True)
    (Path("results") / f"swe_v2_mcp_{instance_id}.json").write_text(
        json.dumps(result, indent=2)
    )

    print()
    print("=" * 50)
    for key, value in result.items():
        print(f"{key}: {value}")


asyncio.run(main())