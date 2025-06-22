from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from dotenv import load_dotenv


class AgenState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a: int, b: int):
    """Add 2 numbers"""
    return a + b


tools = [add]

llm = ChatOllama(
    model="qwen3",
    temperature=0.8,
    num_predict=256,
).bind_tools(tools)


def model_call(state: AgenState) -> AgenState:
    system_prompt = SystemMessage(
        content="You are my coding assistant. Please answeer quentions to the best of your ability."
    )

    response = llm.invoke([system_prompt] + state["messages"])

    return AgenState(messages=[response])


def should_continue(state: AgenState):
    messages = state["messages"]
    last_message = messages[-1]

    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


graph = StateGraph(AgenState)
graph.add_node("model_call", model_call)

tool_node = ToolNode(tools)
graph.add_node("tools", tool_node)

graph.add_edge(START, "model_call")

graph.add_conditional_edges(
    "model_call", should_continue, {"continue": "tools", "end": END}
)

graph.add_edge("tools", "model_call")


app = graph.compile()


def print_stream(stream):
    for s in stream:
        print(s["messages"])
        messages = s["messages"][-1]
        if isinstance(messages, tuple):
            print(messages)

        else:
            messages.pretty_print()


inputs = AgenState(messages=[HumanMessage(content="add 3 + 4, 25+99, 63 + 84")])

# inputs = {"messages": [("user", "Add 3 + 4")]}
print_stream(app.stream(inputs, stream_mode="values"))
