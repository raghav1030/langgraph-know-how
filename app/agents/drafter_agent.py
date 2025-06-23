from typing import TypedDict, Sequence, Annotated
from langgraph.graph import END, StateGraph
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
    BaseMessage,
)
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()

document_content = ""


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def update(content: str) -> str:
    """Updates the document with the provided content."""
    global document_content
    document_content = content
    return f"Document has been updated successfully! The current content is:\n{document_content}."


@tool
def save(filename: str) -> str:
    """Saves the current document to a text file and finish the process."""
    global document_content

    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"

    try:
        with open(filename, "w") as file:
            file.write(document_content)
        print(f"\nDocument has been saved to {filename}")
        return f"\nDocument has been saved successfully to {filename}."
    except Exception as e:
        return f"Error while saving file: {str(e)}"


tools = [update, save]

model = ChatOllama(
    model="qwen3",
    temperature=0.8,
    num_predict=256,
).bind_tools(tools)


def call_model(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content=f"""
You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.

- If the user wants to update or modify content, use the 'update' tool with the complete updated content.
- If the user wants to save and finish, you need to use the 'save' tool.
- Make sure to always show the current document state after modifications.

The current document content is: {document_content}
"""
    )

    if not state["messages"]:
        user_input = "I am ready to update a document. What would you like to create?"
        user_message = AIMessage(content=user_input)
    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\nUSER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)

    print(f"\nAI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}


def should_continue(state: AgentState) -> str:
    """Determine whether to stop or continue the conversation"""

    messages = state["messages"]
    if not messages:
        return "continue"

    for message in reversed(messages):
        # Corrected: Should check message (not messages) and type
        if (
            isinstance(message, ToolMessage)
            and "saved" in message.content.lower()
            and "document" in message.content.lower()
        ):
            return "end"

    return "continue"


def print_statements(messages):
    """Make the statements in a more readable format"""
    if not messages:
        return

    for message in messages[-3:]:  # Show last 3 messages
        if isinstance(message, ToolMessage):
            print(f"TOOL RESULT: {message.content}")
        elif isinstance(message, AIMessage):
            print(f"AI: {message.content}")
        elif isinstance(message, HumanMessage):
            print(f"USER: {message.content}")
        elif isinstance(message, SystemMessage):
            print(f"SYSTEM: {message.content}")


workflow = StateGraph(AgentState)

workflow.add_node("call_model", call_model)
workflow.add_node("tool_node", ToolNode(tools))

workflow.set_entry_point("call_model")
workflow.add_edge("call_model", "tool_node")

workflow.add_conditional_edges(
    "tool_node", should_continue, {"continue": "call_model", "end": END}
)

app = workflow.compile()


def run_document_agent():
    print("==== DRAFTER STARTS ====")

    state = {"messages": []}

    for step in app.stream(state, stream_mode="messages"):
        if "messages" in step:
            print_statements(step["messages"])

    print("==== DRAFTER ENDS ====")


if __name__ == "__main__":
    run_document_agent()
