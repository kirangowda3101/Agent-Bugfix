import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic

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

while step < MAX_STEPS:
    step += 1
    print(f"--- step {step} ---")

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        tools=tools,
        messages=messages,
    )

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
                    for f in Path("workspace").iterdir()
                    if f.is_file() and not f.name.startswith(".")
                ]
                result = "\n".join(files)

            elif block.name == "read_file":
                filename = block.input["filename"]
                path = Path("workspace") / filename
                if path.exists():
                    result = path.read_text()
                else:
                    result = f"Error: {filename} not found"

            elif block.name == "write_file":
                filename = block.input["filename"]
                content = block.input["content"]
                path = Path("workspace") / filename
                if path.resolve().parent != Path("workspace").resolve():
                    result = "Error: cannot write outside workspace"
                else:
                    path.write_text(content)
                    result = f"Wrote {len(content)} characters to {filename}"

            elif block.name == "run_tests":
                try:
                    proc = subprocess.run(
                        [
                            "docker", "run", "--rm",
                            "-v", f"{Path('workspace').resolve()}:/app",
                            "-w", "/app",
                            "bugfix-sandbox",
                            "timeout", "25", "python", "-m", "pytest", "-q",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    result = f"exit code: {proc.returncode}\n" + proc.stdout + proc.stderr
                except subprocess.TimeoutExpired:
                    result = "TIMEOUT: tests took longer than 30 seconds"

            else:
                result = f"Error: unknown tool {block.name}"

            print("TOOL RESULT:", result)

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

    messages.append({"role": "user", "content": tool_results})

print()
print("steps used:", step, "of", MAX_STEPS)
print("messages length:", len(messages))