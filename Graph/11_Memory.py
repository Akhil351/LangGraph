from llm import get_open_ai_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from typing import TypedDict, Annotated, List
from langchain_core.messages import HumanMessage

# ---------------- STATE ---------------- #
class State(TypedDict):
    messages: Annotated[List, add_messages]

# ---------------- LLM ---------------- #
llm = get_open_ai_model()

# ---------------- NODE ---------------- #
def chatbot_node(state: State) -> State:
    response = llm.invoke(state["messages"])
    return {
        "messages": [response]
    }

# ---------------- GRAPH ---------------- #
builder = StateGraph(State)

builder.add_node("chatbot", chatbot_node)

builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()


state = {"messages": []}

print("💬 Chat started (type 'exit' to stop)\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("👋 Chat ended")
        break

    # Add user message
    state["messages"].append(HumanMessage(content=user_input))

    # Call graph
    state = graph.invoke(state)

    # Get last AI response
    ai_response = state["messages"][-1].content

    print("AI:", ai_response)