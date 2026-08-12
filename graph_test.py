from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    messages: list
    count: int


def alice(state: State) -> dict:
    print("alice is running")
    return {
        "messages": state["messages"] + ["alice says hello"],
        "count": state["count"] + 1,
    }


def bob(state: State) -> dict:
    print("bob is running")
    print("bob received:", state["messages"])
    return {
        "messages": state["messages"] + ["bob says goodbye"],
        "count": state["count"] + 1,
    }


builder = StateGraph(State)

builder.add_node("alice", alice)
builder.add_node("bob", bob)

builder.add_edge(START, "alice")
builder.add_edge("alice", "bob")
builder.add_edge("bob", END)

graph = builder.compile()

result = graph.invoke({"messages": [], "count": 0})

print()
print("final messages:", result["messages"])
print("final count:", result["count"])