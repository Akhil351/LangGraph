from llm import get_open_ai_model
from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict, Literal
from pydantic import BaseModel, Field
from operator import add

# ---------------- SCHEMA ---------------- #
class LLMSchema(BaseModel):
    category: Literal["instagram", "linkedin", "twitter"] = Field(
        description="Category of the post to generate"
    )
    topic: str = Field(
        description="Topic of the post to generate"
    )

# ---------------- LLMs ---------------- #
router_llm = get_open_ai_model().with_structured_output(LLMSchema)
content_llm = get_open_ai_model()

# ---------------- STATE ---------------- #
class State(TypedDict):
    input: str
    topic: str
    post: Annotated[str, add]
    category: str

# ---------------- NODES ---------------- #
def create_post_instagram(state: State) -> State:
    topic = state["topic"]
    response = content_llm.invoke(
        f"Create an Instagram post about {topic}. Keep it under 50 words."
    ).content
    return {"post": response}


def create_post_linkedin(state: State) -> State:
    topic = state["topic"]
    response = content_llm.invoke(
        f"Create a LinkedIn post about {topic}. Keep it under 200 words."
    ).content
    return {"post": response}


def create_post_twitter(state: State) -> State:
    topic = state["topic"]
    response = content_llm.invoke(
        f"Create a Twitter post about {topic}. Keep it under 280 characters."
    ).content
    return {"post": response}


def decided_node(state: State) -> State:
    input_text = state["input"]
    response = router_llm.invoke(input_text)

    return {
        "category": response.category,
        "topic": response.topic
    }


def decided_next_node(state: State) -> str:
    category = state["category"]

    if category == "instagram":
        return "Instagram_Node"
    elif category == "linkedin":
        return "LinkedIn_Node"
    elif category == "twitter":
        return "Twitter_Node"
    else:
        raise ValueError("Invalid category")


# ---------------- GRAPH ---------------- #
graph = StateGraph(State)

graph.add_node("Instagram_Node", create_post_instagram)
graph.add_node("LinkedIn_Node", create_post_linkedin)
graph.add_node("Twitter_Node", create_post_twitter)
graph.add_node("Decider_Node", decided_node)

graph.add_edge(START, "Decider_Node")

graph.add_conditional_edges(
    "Decider_Node",
    decided_next_node,
    {
        "Instagram_Node": "Instagram_Node",
        "LinkedIn_Node": "LinkedIn_Node",
        "Twitter_Node": "Twitter_Node",
    },
)

graph.add_edge("Instagram_Node", END)
graph.add_edge("LinkedIn_Node", END)
graph.add_edge("Twitter_Node", END)

graph = graph.compile()

# ---------------- RUN ---------------- #
result = graph.invoke({
    "input": "I want to create a post about the impact of AI in LinkedIn."
})

print(result["post"])