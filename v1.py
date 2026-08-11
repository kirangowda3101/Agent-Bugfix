import os
import shutil
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic

problem = sys.argv[1]

workspace = Path("workspace")
source = Path("dataset") / problem

shutil.rmtree(workspace, ignore_errors=True)
shutil.copytree(source / "buggy", workspace)
shutil.copy(source / "tests" / "test_solution.py", workspace)


def run_pytest():
    try:
        proc = subprocess.run(
            [
                "docker", "run", "--rm",
                "-v", f"{workspace.resolve()}:/app",
                "-w", "/app",
                "bugfix-sandbox",
                "timeout", "25", "python", "-m", "pytest", "-q",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return proc.returncode, f"exit code: {proc.returncode}\n" + proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT: tests took longer than 30 seconds"


load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

tools = [
    {
        "name": "list_files",
        "description": "List all files in the working directory.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file in the working directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "The name of the file to read, e.g. solution.py",
                }
            },
            "required": ["filename"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file in the working directory. Overwrites the file completely.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "The name of the file to write, e.g. helpers.py",
                },
                "content": {
                    "type": "string",
                    "description": "The complete new contents of the file",
                },
            },
            "required": ["filename", "content"],
        },
    },
    {
        "name": "run_tests",
        "description": "Run the test suite in the working directory and return the results. Use this to verify whether your fix actually works before finishing.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

messages = [
    {
        "role": "user",
        "content": "The tests are failing in the working directory. Find and fix the bug.",
    }
]

MAX_STEPS = 10
step = 0
input_tokens = 0
output_tokens = 0

while step < MAX_STEPS:
    step += 1
    print(f"--- step {step} ---")

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
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

            if block.name == "list_files":
                files = [
                    f.name
                    for f in workspace.iterdir()
                    if f.is_file() and not f.name.startswith(".")
                ]
                result = "\n".join(files)

            elif block.name == "read_file":
                filename = block.input["filename"]
                path = workspace / filename
                if path.exists():
                    result = path.read_text()
                else:
                    result = f"Error: {filename} not found"

            elif block.name == "write_file":
                filename = block.input["filename"]
                content = block.input["content"]
                path = workspace / filename
                if path.resolve().parent != workspace.resolve():
                    result = "Error: cannot write outside workspace"
                else:
                    path.write_text(content)
                    result = f"Wrote {len(content)} characters to {filename}"

            elif block.name == "run_tests":
                _, result = run_pytest()

            else:
                result = f"Error: unknown tool {block.name}"

            print("TOOL RESULT:", result)

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

    messages.append({"role": "user", "content": tool_results})

returncode, final_output = run_pytest()

print()
print("=" * 50)
print("problem:", problem)
print("verdict:", "PASS" if returncode == 0 else "FAIL")
print("steps used:", step, "of", MAX_STEPS)
print("input tokens:", input_tokens)
print("output tokens:", output_tokens)
print()
print(final_output)