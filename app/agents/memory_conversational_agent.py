from typing import TypedDict, List, Union
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]


llm = ChatOllama(
    model="qwen3",
    temperature=0.8,
    num_predict=256,
)


def process(state: AgentState) -> AgentState:
    """this function process the input given by human

    Args:
        state (AgentState): _description_

    Returns:
        AgentState: _description_
    """
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}")


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
app = graph.compile()

conversation_history = []
user_input = input("Ask: \t")


while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = app.invoke(AgentState(messages=conversation_history))
    print(result)
    conversation_history = result["messages"]
    user_input = input("Ask: \t")
