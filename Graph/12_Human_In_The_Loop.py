from llm import get_open_ai_model
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage


llm = get_open_ai_model()

# ---------------- STATE ---------------- #
class State(TypedDict):
    messages: Annotated[List, add_messages]
    approved: bool


# ---------------- NODE 1: LLM ---------------- #
def chatbot_node(state: State) -> State:
    user_msg = state["messages"][-1].content
    
    response = llm.invoke(state["messages"])
    
    return {
        "messages": [AIMessage(content=response.content)] 
    }


# ---------------- NODE 2: HUMAN APPROVAL ---------------- #
def human_node(state: State) -> State:
    print("\nAI says:", state["messages"][-1].content)
    
    approval = input("Approve this response? (yes/no): ")
    
    return {
        "approved": approval.lower() == "yes"
    }


# ---------------- NODE 3: FINAL ---------------- #
def final_node(state: State) -> State:
    if state["approved"]:
        print("\n✅ Final Response Sent:", state["messages"][-1].content)
    else:
        print("\n❌ Response Rejected by Human")

    return state


# ---------------- GRAPH ---------------- #
builder = StateGraph(State)

builder.add_node("chatbot", chatbot_node)
builder.add_node("human", human_node)
builder.add_node("final", final_node)

builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", "human")
builder.add_edge("human", "final")
builder.add_edge("final", END)

graph = builder.compile()


# ---------------- RUN ---------------- #
state = {
    "messages": [HumanMessage(content="What is AI?")],
    "approved": False
}

graph.invoke(state)