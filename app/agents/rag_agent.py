from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
)
from operator import add as add_messages
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import GithubFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool

load_dotenv()

llm = ChatOllama(
    model="qwen3",
    temperature=0.8,
    num_predict=256,
)


embeddings = OllamaEmbeddings(
    model="dengcao/Qwen3-Embedding-4B:Q4_K_M",
)


access_token = os.getenv("GITHUB_ACCESS_TOKEN")

loader = GithubFileLoader(
    access_token=access_token,
    repo="raghav1030/NaboServer",
    branch="main",  # <--- Change from "master" to "main"
    github_api_url="https://api.github.com",
    file_filter=None,
)


# Checks if the PDF is there
try:
    pages = loader.load()
    print(f"Github Files have been loaded and has {len(pages)} pages")
except Exception as e:
    print(f"Error loading Github Files: {e}")
    raise

# Chunking Process
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


pages_split = text_splitter.split_documents(pages)  # We now apply this to our pages

persist_directory = r"C:\Users\Raghav\Desktop\Langraph\app\agents"
collection_name = "github_repo"

# If our collection does not exist in the directory, we create using the os command
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)


try:
    # Here, we actually create the chroma database using our embeddigns model
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    print(f"Created ChromaDB vector store!")

except Exception as e:
    print(f"Error setting up ChromaDB: {str(e)}")
    raise


# Now we create our retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},  # K is the amount of chunks to return
)


@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches the GitHub repository for relevant file contents based on your query.
    Use this tool to retrieve and reference the actual content from files in the repository.
    """

    docs = retriever.invoke(query)

    if not docs:
        return "No relevant content found in the GitHub repository for your query."

    results = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown file")
        results.append(f"File: {source}\nContent:\n{doc.page_content}")

    return "\n\n".join(results)


tools = [retriever_tool]

llm = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state["messages"][-1]
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0


system_prompt = """
You are an intelligent AI assistant that answers questions about the contents of a GitHub repository.
Use the retriever tool to search and reference files from the repository as needed to answer user questions.
Always cite the specific file names and relevant content you use in your answers.
If you need more information, you may make multiple retrievals.
Be concise, accurate, and helpful.
"""


tools_dict = {
    our_tool.name: our_tool for our_tool in tools
}  # Creating a dictionary of our tools


# LLM Agent
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {"messages": [message]}


# Retriever Agent
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state["messages"][-1].tool_calls
    results = []
    for t in tool_calls:
        print(
            f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}"
        )

        if not t["name"] in tools_dict:  # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."

        else:
            result = tools_dict[t["name"]].invoke(t["args"].get("query", ""))
            print(f"Result length: {len(str(result))}")

        # Appends the Tool Message
        results.append(
            ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result))
        )

    print("Tools Execution Complete. Back to the model!")
    return {"messages": results}


graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm", should_continue, {True: "retriever_agent", False: END}
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()


def running_agent():
    print("\n=== RAG AGENT===")

    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        messages = [
            HumanMessage(content=user_input)
        ]  # converts back to a HumanMessage type

        result = rag_agent.invoke({"messages": messages})

        print("\n=== ANSWER ===")
        print(result["messages"][-1].content)


running_agent()
