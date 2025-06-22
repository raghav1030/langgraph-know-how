from typing import TypedDict, List
from langgraph.graph import StateGraph
from IPython.display import display, Image

class AgentState(TypedDict):
    values: List[int]
    name: str
    result: str


def process_values(state: AgentState) -> AgentState:
    """This function handles multiple different values

    Args:
        state (AgentState): _description_

    Returns:
        AgentState: _description_
    """


    state['result'] = f"Hi there {state['name']}! Your sum of numbers {state['values']} is {sum(state['values'])}"
    return state

graph = StateGraph(AgentState)

graph.add_node("processor", process_values)
graph.set_entry_point("processor")
graph.set_finish_point("processor")

app = graph.compile()

display(Image(app.get_graph().draw_mermaid_png()))
answers = app.invoke({"name" : "Raghav", "values": [1,2]})

print(answers)
