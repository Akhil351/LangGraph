from llm import get_open_ai_model
from langgraph.graph import StateGraph,START,END
from pydantic import BaseModel, Field


llm=get_open_ai_model()

class State(BaseModel):
    topic: str= Field(description="Topic Name")
    explanation: str= Field(default="",description="Detailed explanation")
    summary: str= Field(default="",description="SShort Summary")



def generate_explanation(state: State)-> State:
    # convert pydantic model to dict ( state=state.model_dump())
    topic=state.topic
    explanation=llm.invoke(f"Explain the topic {topic} in detail").content
    state.explanation=explanation
    return state


def generate_summary(state: State)-> State:
    explanation=state.explanation
    summary=llm.invoke(f"Summarize the following explanation in short:{explanation}").content
    state.summary=summary
    return state

builder=StateGraph(State)

builder.add_node("generate_explanation",generate_explanation)
builder.add_node("generate_summary",generate_summary)

builder.add_edge(START,"generate_explanation")
builder.add_edge("generate_explanation","generate_summary")
builder.add_edge("generate_summary",END)

graph=builder.compile()

response=graph.invoke({"topic":"Artificial Intelligence"})

print(response)
