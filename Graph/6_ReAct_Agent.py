from llm import get_open_ai_model
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain.tools import tool
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from operator import add
from core import settings


# ---------------- STATE ---------------- #
class State(TypedDict):
    messages: Annotated[
        list[SystemMessage | HumanMessage | AIMessage | ToolMessage],
        add,
    ]


# ---------------- TOOL SETUP ---------------- #
search = SerpAPIWrapper(
    serpapi_api_key=settings["SERPAPI_API_KEY"]
)


@tool
def google_search(query: str) -> str:
    """Search Google for the latest information"""
    result = search.run(query)
    return result


tools = [google_search]

# Map tool name → tool function
tools_by_name = {tool.name: tool for tool in tools}


# ---------------- LLM ---------------- #
llm = get_open_ai_model().bind_tools(tools)


# ---------------- LLM NODE ---------------- #
def llm_agent_node(state: State) -> State:
    messages = state["messages"]

    response = llm.invoke(messages)

    return {
        "messages": [response]
    }


# ---------------- TOOL NODE ---------------- #
def tool_node(state: State) -> State:
    messages = state["messages"]
    tool_results = []

    last_message = messages[-1]

    # Safety check
    if not getattr(last_message, "tool_calls", None):
        return {"messages": []}

    for tool_call in last_message.tool_calls:
        tool = tools_by_name[tool_call["name"]]

        observation = str(tool.invoke(tool_call["args"]))

        tool_results.append(
            ToolMessage(
                content=observation,
                tool_call_id=tool_call["id"]
            )
        )

    return {"messages": tool_results}


# ---------------- CONDITION NODE ---------------- #
def decide_next_node(state: State) -> str:
    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "Tool_Node"
    else:
        return "end"


# ---------------- GRAPH ---------------- #
graph = StateGraph(State)

graph.add_node("LLM_Agent_Node", llm_agent_node)
graph.add_node("Tool_Node", tool_node)

graph.add_edge(START, "LLM_Agent_Node")

graph.add_conditional_edges(
    "LLM_Agent_Node",
    decide_next_node,{
        "Tool_Node": "Tool_Node",
        "end": END
    }
)

graph.add_edge("Tool_Node", "LLM_Agent_Node")

# Compile graph
react_graph = graph.compile()



result=react_graph.invoke({"messages":[SystemMessage(content="You are a helpful assistant that can use tools to answer questions"), HumanMessage(content="What is the latest news about the Iran–U.S. war?")]})


for msg in result["messages"]:
    if isinstance(msg, SystemMessage):
        print(f"🟢 SYSTEM: {msg.content}")

    elif isinstance(msg, HumanMessage):
        print(f"👤 HUMAN: {msg.content}")

    elif isinstance(msg, AIMessage):
        print(f"🤖 AI: {msg.content}")

        # Optional: show tool calls
        if msg.tool_calls:
            print(f"   🔧 Tool Calls: {msg.tool_calls}")

    elif isinstance(msg, ToolMessage):
        print(f"🛠️ TOOL RESULT: {msg.content}")

    print("-" * 60)