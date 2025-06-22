from typing import TypedDict
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()


class AgentState(TypedDict):
    messages: list[HumanMessage]


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
    print(response.content)


graph = StateGraph(AgentState)
graph.add_node("process",process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
app = graph.compile()

user_input = input("Ask: \t")


while user_input != 'exit':

    app.invoke(AgentState(messages=[HumanMessage(content=user_input)]))
    user_input = input("Ask: \t")
