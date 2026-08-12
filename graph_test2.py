from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    log: list
    next: str
    steps: int


MAX_STEPS = 8


def supervisor(state: State) -> dict:
    steps = state["steps"] + 1

    if steps > MAX_STEPS:
        decision = "done"
    elif "researcher done" not in state["log"]:
        decision = "researcher"
    elif "coder done" not in state["log"]:
        decision = "coder"
    else:
        decision = "done"

    print(f"[supervisor] step {steps}, sending to: {decision}")
    return {"next": decision, "steps": steps}


def researcher(state: State) -> dict:
    print("  [researcher] working")
    return {"log": state["log"] + ["researcher done"]}


def coder(state: State) -> dict:
    print("  [coder] working")
    return {"log": state["log"] + ["coder done"]}


def route(state: State) -> str:
    return state["next"]


builder = StateGraph(State)

builder.add_node("supervisor", supervisor)
builder.add_node("researcher", researcher)
builder.add_node("coder", coder)

builder.add_edge(START, "supervisor")

builder.add_conditional_edges(
    "supervisor",
    route,
    {
        "researcher": "researcher",
        "coder": "coder",
        "done": END,
    },
)

builder.add_edge("researcher", "supervisor")
builder.add_edge("coder", "supervisor")

graph = builder.compile()

result = graph.invoke({"log": [], "next": "", "steps": 0})

print()
print("log:", result["log"])
print("steps used:", result["steps"])