from llm import get_open_ai_model
from langchain_community.utilities import SerpAPIWrapper
from langgraph.graph import StateGraph, START, END
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from typing import Annotated, TypedDict
from operator import add
from core import settings

# ---------------- STATE ---------------- #
class State(TypedDict):
    messages: Annotated[list[SystemMessage | HumanMessage | AIMessage | ToolMessage], add]

# ---------------- TOOL ---------------- #
search = SerpAPIWrapper(serpapi_api_key=settings["SERPAPI_API_KEY"])

@tool
def google_search(query: str) -> str:
    """Search google for the latest information"""
    result = search.run(query)
    return result

tools = [google_search]

# ---------------- LLM ---------------- #
llm = get_open_ai_model().bind_tools(tools)

# ---------------- AGENT NODE ---------------- #
def agent_node(state: State) -> State:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# ---------------- DECISION ---------------- #
def decide_next_node(state: State):
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    else:
        return END

# ---------------- GRAPH ---------------- #
builder = StateGraph(State)

builder.add_node("agent_node", agent_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent_node")

builder.add_conditional_edges(
    "agent_node",
    decide_next_node,
    {
        "tools": "tools",
        END: END
    }
)

builder.add_edge("tools", "agent_node")

graph = builder.compile()

messages=[SystemMessage(content="You are a helpful  assistant with access to google search.")]


while True:
    user_input = input("User: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Bot: GoodBye!")
        break

    messages.append(HumanMessage(content=user_input))

    result = graph.invoke({"messages": messages})


    messages = result["messages"]

    last_message = messages[-1]

    print(f"Bot: {last_message.content}")




