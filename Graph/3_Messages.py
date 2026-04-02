from llm import get_open_ai_model
from langgraph.graph import StateGraph, START,END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import Annotated, TypedDict
from  operator import add
llm=get_open_ai_model()


class State(TypedDict):
    messages: Annotated[list[SystemMessage| HumanMessage| AIMessage],add]


def node(state: State)-> State:
    response= llm.invoke(state["messages"])
    state["messages"]=[AIMessage(content=response.content)]
    return state


builder=StateGraph(State)

builder.add_node("node",node)
builder.add_edge(START,"node")
builder.add_edge("node",END)

graph=builder.compile()

response=graph.invoke({"messages":[HumanMessage(content="Hi my name is Akhil")]})

for message in response["messages"]:
   print(f"{message.__class__.__name__}: {message.content}")



