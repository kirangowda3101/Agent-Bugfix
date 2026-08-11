import os
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
]

messages = [
    {
        "role": "user",
        "content": "The tests are failing in the working directory. Find and fix the bug.",
    }
]

while True:
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
print("messages length:", len(messages))