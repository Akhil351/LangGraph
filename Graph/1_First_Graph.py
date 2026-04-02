from llm import get_open_ai_model
from langgraph.graph import StateGraph, START, END
from typing import TypedDict


# Define state
class State(TypedDict):
    input: str
    output: str

# Load model
llm = get_open_ai_model()

#Define node
def node(state: State)->State:
    response=llm.invoke(state["input"])
    state["output"]=response.content
    return state


# Build Graph
builder=StateGraph(State)
builder.add_node("node",node)
builder.add_edge(START, "node")
builder.add_edge("node", END)

graph=builder.compile()

response=graph.invoke({"input":"My name is Akhil"})

print(response)

    
