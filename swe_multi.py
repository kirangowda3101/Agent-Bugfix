import asyncio
import json
import os
import subprocess
import sys
from typing import TypedDict
from pathlib import Path
from datasets import load_dataset
from dotenv import load_dotenv
from anthropic import Anthropic
from mcp import Client
from langgraph.graph import StateGraph, START, END
import mcp_server

instance_id = sys.argv[1]

ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
instance = next((r for r in ds if r["instance_id"] == instance_id), None)
if instance is None:
    sys.exit(f"instance {instance_id} not found")

repo_part, name_part = instance_id.split("__", 1)
image = f"swebench/sweb.eval.x86_64.{repo_part}_1776_{name_part}:latest"
container_name = f"swe_agent_{instance_id}"

subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
subprocess.run(
    ["docker", "run", "-d", "--name", container_name, image, "sleep", "3600"],
    check=True,
)
mcp_server.set_container(container_name)

load_dotenv()
anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODEL = "claude-haiku-4-5-20251001"
MAX_STEPS = 12
WORKER_MAX_TURNS = 6

ROLE_TOOLS = {
    "researcher": ["search_code", "read_file"],
    "coder": ["read_file", "write_file"],
    "tester": ["run_tests"],
}

ROLE_INSTRUCTIONS = {
    "researcher": "You are the RESEARCHER. Find the code responsible for the bug. Do not attempt to fix it. When done, state clearly which file and which lines are the problem.",
    "coder": "You are the CODER. Apply a fix based on the findings provided. Read the file first, then write the corrected version. Do not run tests.",
    "tester": "You are the TESTER. Run the relevant Django test module and report exactly what passed or failed.",
}


class State(TypedDict):
    findings: list
    changes: list
    test_output: list
    next: str
    steps: int
    input_tokens: int
    output_tokens: int


mcp_client = None
all_tools = []


def tools_for(role):
    names = ROLE_TOOLS[role]
    return [t for t in all_tools if t["name"] in names]


def context_block(state):
    parts = [f"GITHUB ISSUE:\n{instance['problem_statement']}"]
    if state["findings"]:
        parts.append("RESEARCHER FINDINGS:\n" + "\n".join(state["findings"]))
    if state["changes"]:
        parts.append("CHANGES MADE:\n" + "\n".join(state["changes"]))
    if state["test_output"]:
        parts.append("TEST RESULTS:\n" + "\n".join(state["test_output"]))
    return "\n\n".join(parts)


async def run_worker(role, state):
    messages = [
        {
            "role": "user",
            "content": ROLE_INSTRUCTIONS[role]
            + "\n\nThe repository is at /testbed.\n\n"
            + context_block(state),
        }
    ]

    tools = tools_for(role)
    turns = 0
    used_in = 0
    used_out = 0
    summary = ""

    while turns < WORKER_MAX_TURNS:
        turns += 1
        response = anthropic.messages.create(
            model=MODEL, max_tokens=4000, tools=tools, messages=messages
        )
        used_in += response.usage.input_tokens
        used_out += response.usage.output_tokens

        for block in response.content:
            if block.type == "text":
                summary = block.text
                print(f"    [{role}] {block.text[:200]}")

        if response.stop_reason != "tool_use":
            break

        messages.append({"role": "assistant", "content": response.content})

        results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"    [{role}] TOOL: {block.name} {block.input}")
                r = await mcp_client.call_tool(block.name, block.input)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": r.content[0].text,
                })
        messages.append({"role": "user", "content": results})

    return summary or f"({role} produced no summary)", used_in, used_out


async def supervisor(state):
    steps = state["steps"] + 1

    if steps > MAX_STEPS:
        print(f"[supervisor] step {steps}: STEP BUDGET EXCEEDED")
        return {"next": "done", "steps": steps}

    prompt = (
        "You are the SUPERVISOR of a bug-fixing team. Decide who works next.\n\n"
        "researcher - finds the buggy code (no findings yet? start here)\n"
        "coder - applies a fix (needs findings first)\n"
        "tester - runs the tests (needs a change first)\n"
        "done - the tests pass, or no further progress is possible\n\n"
        + context_block(state)
        + "\n\nReply with exactly one word: researcher, coder, tester, or done."
    )

    response = anthropic.messages.create(
        model=MODEL,
        max_tokens=20,
        messages=[{"role": "user", "content": prompt}],
    )

    decision = response.content[0].text.strip().lower()
    if decision not in ("researcher", "coder", "tester", "done"):
        print(f"[supervisor] unparseable decision {decision!r}, stopping")
        decision = "done"

    print(f"[supervisor] step {steps} -> {decision}")
    return {
        "next": decision,
        "steps": steps,
        "input_tokens": state["input_tokens"] + response.usage.input_tokens,
        "output_tokens": state["output_tokens"] + response.usage.output_tokens,
    }


async def researcher_node(state):
    summary, i, o = await run_worker("researcher", state)
    return {
        "findings": state["findings"] + [summary],
        "input_tokens": state["input_tokens"] + i,
        "output_tokens": state["output_tokens"] + o,
    }


async def coder_node(state):
    summary, i, o = await run_worker("coder", state)
    return {
        "changes": state["changes"] + [summary],
        "input_tokens": state["input_tokens"] + i,
        "output_tokens": state["output_tokens"] + o,
    }


async def tester_node(state):
    summary, i, o = await run_worker("tester", state)
    return {
        "test_output": state["test_output"] + [summary],
        "input_tokens": state["input_tokens"] + i,
        "output_tokens": state["output_tokens"] + o,
    }


def route(state):
    return state["next"]


builder = StateGraph(State)
builder.add_node("supervisor", supervisor)
builder.add_node("researcher", researcher_node)
builder.add_node("coder", coder_node)
builder.add_node("tester", tester_node)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    route,
    {"researcher": "researcher", "coder": "coder", "tester": "tester", "done": END},
)
builder.add_edge("researcher", "supervisor")
builder.add_edge("coder", "supervisor")
builder.add_edge("tester", "supervisor")

graph = builder.compile()


async def main():
    global mcp_client, all_tools
    try:
        async with Client(mcp_server.mcp) as client:
            mcp_client = client
            tools_result = await client.list_tools()
            all_tools = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in tools_result.tools
            ]

            final = await graph.ainvoke(
                {
                    "findings": [],
                    "changes": [],
                    "test_output": [],
                    "next": "",
                    "steps": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                },
                config={"recursion_limit": 60},
            )

        patch = mcp_server.run_in_container("git -C /testbed diff")
    finally:
        subprocess.run(["docker", "stop", container_name], capture_output=True)
        subprocess.run(["docker", "rm", container_name], capture_output=True)

    Path("predictions").mkdir(exist_ok=True)
    (Path("predictions") / f"swe_multi_{instance_id}.jsonl").write_text(
        json.dumps({
            "instance_id": instance_id,
            "model_name_or_path": "swe_multi",
            "model_patch": patch,
        }) + "\n"
    )

    result = {
        "version": "swe_multi",
        "instance_id": instance_id,
        "steps": final["steps"],
        "input_tokens": final["input_tokens"],
        "output_tokens": final["output_tokens"],
        "patch_length": len(patch),
    }
    Path("results").mkdir(exist_ok=True)
    (Path("results") / f"swe_multi_{instance_id}.json").write_text(
        json.dumps(result, indent=2)
    )

    print()
    print("=" * 50)
    for k, v in result.items():
        print(f"{k}: {v}")


asyncio.run(main())