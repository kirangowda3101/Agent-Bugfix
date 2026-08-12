import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datasets import load_dataset
from dotenv import load_dotenv
from anthropic import Anthropic
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

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

subprocess.run(
    ["git", "checkout", instance["base_commit"]],
    cwd="repos/django",
    check=True,
)
print(f"Checked out {instance['base_commit']} in repos/django")

print("Building FAISS index from repos/django ...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

texts = []
metadatas = []
file_count = 0
for path in Path("repos/django").glob("**/*.py"):
    try:
        content = path.read_text(errors="ignore")
    except Exception:
        continue
    file_count += 1
    rel_path = str(path.relative_to("repos/django"))
    for chunk in splitter.split_text(content):
        texts.append(chunk)
        metadatas.append({"path": rel_path})

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

build_start = time.time()
store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
index_build_seconds = time.time() - build_start

chunk_count = len(texts)
print(f"Indexed {file_count} files into {chunk_count} chunks in {index_build_seconds:.1f}s")


def run_in_container(cmd):
    proc = subprocess.run(
        ["docker", "exec", "-w", "/testbed", container_name, "bash", "-c", cmd],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return proc.stdout + proc.stderr


def search_code(query):
    results = store.similarity_search(query, k=5)
    parts = []
    for doc in results:
        parts.append(f"{doc.metadata['path']}\n{doc.page_content}")
    output = "\n\n".join(parts)
    if len(output) > 6000:
        output = output[:6000] + "\n... (truncated)"
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


def run_tests(test_module):
    output = run_in_container(
        "source /opt/miniconda3/bin/activate && conda activate testbed && "
        f"./tests/runtests.py --settings=test_sqlite {test_module}"
    )
    lines = output.splitlines()
    return "\n".join(lines[-50:])


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
        "description": "Search the codebase semantically. Describe what you are looking for in natural language, e.g. 'function that parses duration strings'. Returns the most relevant code sections.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language description of the code you are looking for, e.g. function that parses duration strings",
                }
            },
            "required": ["query"],
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
    {
        "name": "run_tests",
        "description": "Run a Django test module and return the results. Use this to verify whether your fix actually works before finishing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "test_module": {
                    "type": "string",
                    "description": "The test module to run, e.g. utils_tests.test_dateparse",
                }
            },
            "required": ["test_module"],
        },
    },
]

prompt = (
    "You are working in a Django repository checked out at /testbed inside a Docker container.\n\n"
    f"{instance['problem_statement']}\n\n"
    "Locate the bug described above and fix it using the available tools. "
    "When you believe the fix is complete, run the relevant tests to confirm."
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
                    result = search_code(block.input["query"])

                elif block.name == "read_file":
                    result = read_file(block.input["filename"])

                elif block.name == "write_file":
                    result = write_file(block.input["filename"], block.input["content"])

                elif block.name == "run_tests":
                    result = run_tests(block.input["test_module"])

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
    "model_name_or_path": "swe_v2_rag",
    "model_patch": patch,
}
(Path("predictions") / f"swe_v2_rag_{instance_id}.jsonl").write_text(json.dumps(prediction) + "\n")

result = {
    "version": "swe_v2_rag",
    "instance_id": instance_id,
    "steps": step,
    "input_tokens": input_tokens,
    "output_tokens": output_tokens,
    "patch_length": len(patch),
    "index_build_seconds": round(index_build_seconds, 1),
    "chunk_count": chunk_count,
}

Path("results").mkdir(exist_ok=True)
(Path("results") / f"swe_v2_rag_{instance_id}.json").write_text(json.dumps(result, indent=2))

print()
print("=" * 50)
for key, value in result.items():
    print(f"{key}: {value}")
