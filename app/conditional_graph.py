from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class AgentState(TypedDict):
    number1: int
    operation: str
    number2: int
    finalNumber: int


def adder(state: AgentState) -> AgentState:
    """This node adds two number

    Args:
        state (AgentState): _description_

    Returns:
        AgentState: _description_
    """
    state["finalNumber"] = state["number1"] + state["number2"]

    return state


def subtractor(state: AgentState) -> AgentState:
    """This node subtracts two number

    Args:
        state (AgentState): _description_

    Returns:
        AgentState: _description_
    """
    state["finalNumber"] = state["number1"] - state["number2"]

    return state


def decideNextNode(state: AgentState) -> AgentState:
    """This node decides the next node of the graph

    Args:
        state (AgentState): _description_

    Returns:
        AgentState: _description_
    """
    if state["operation"] == "+":
        return "add_operation"

    elif state["operation"] == "-":
        return "subtract_operation"

    return state


graph = StateGraph(AgentState)

graph.add_node("add_node", adder)
graph.add_node("subtract_node", subtractor)
graph.add_node("router", lambda state: state)

graph.add_edge(START, "router")
graph.add_conditional_edges(
    "router",
    decideNextNode,
    {"add_operation": "add_node", "subtract_operation": "subtract_node"},
)

graph.add_edge("add_node", END)
graph.add_edge("subtract_node", END)


app = graph.compile()

inital_state_1 = AgentState(number1=10, number2=20, operation="+")
print(app.invoke(inital_state_1))
inital_state_2 = AgentState(number1=30, number2=20, operation="-")
print(app.invoke(inital_state_2))
