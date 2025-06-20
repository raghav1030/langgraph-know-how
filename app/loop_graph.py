from typing import TypedDict, List
from langgraph.graph import StateGraph, END, START
import random


class AgentState(TypedDict):
    name: str
    number: List[int]
    counter: int


def greeting_node(state: AgentState) -> AgentState:
    """This function greets the person

    Args:
        state (AgentState): _description_

    Returns:
        AgentState: _description_
    """
    state["name"] = f"Hi there, {state['name']}"
    state["counter"] = 0
    return state


def random_node(state: AgentState) -> AgentState:
    """This is a random function that generates a number from 0 to 10

    Args:
        state (AgentState): _description_

    Returns:
        AgentState: _description_
    """

    state["number"].append(random.randint(0, 10))
    state["counter"] += 1

    return state


def should_continue(state: AgentState) -> AgentState:
    """Function that tels whether to continue looping or break"""

    if state["counter"] < 5:
        print(f"ENTERING LOOP {state['counter']}")
        return "loop"
    else:
        return "exit"


graph = StateGraph(AgentState)

graph.add_node("greeting_node", greeting_node)
graph.add_node("random_node", random_node)
graph.add_edge(START, "greeting_node")
graph.add_edge("greeting_node", "random_node")

graph.add_conditional_edges(
    "random_node", should_continue, {"loop": "random_node", "exit": END}
)


app = graph.compile()

state = AgentState(name="Raghav", number=[])

print(app.invoke(state))
