from typing import TypedDict
from langgraph.graph import StateGraph


class AgentState(TypedDict):
    name: str
    age: str
    final: str


def first_node(state: AgentState) -> AgentState:
    """First node of our sequence

    Args:
        state (AgentState): First node of our sequence state

    Returns:
        AgentState: First node of our sequence state
    """

    state["final"] = f"Hi {state['name']}!"
    return state


def second_node(state: AgentState) -> AgentState:
    """second node of our sequence

    Args:
        state (AgentState): second node of our sequence state

    Returns:
        AgentState: second node of our sequence state
    """

    state["final"] = state["final"] + f" You are {state['age']} years old!"
    return state


graph = StateGraph(AgentState)

graph.add_node("first_node", first_node)
graph.add_node("second_node", second_node)

graph.add_edge("first_node", "second_node")
graph.set_entry_point("first_node")
graph.set_finish_point("second_node")

app = graph.compile()

result = app.invoke({"name": "Raghav", "age": 22})

print(result)
