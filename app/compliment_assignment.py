from typing import TypedDict
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    message: str


def compliment(state: AgentState) -> AgentState:
    """Compliments the name given in the agent state

    Args:
        state (AgentState): _description_

    Returns:
        AgentState: _description_
    """
    state['message'] = "Hey "+ state['message'] + ", you'll make it through. Keep Learning. All the best."
    
    return state

graph = StateGraph(AgentState)

graph.add_node("compliment", compliment)

graph.set_entry_point("compliment")
graph.set_finish_point("compliment")

app = graph.compile()
result = app.invoke({"message": "Raghav"})

print(result['message'])