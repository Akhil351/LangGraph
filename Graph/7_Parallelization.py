from llm import get_open_ai_model
from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict
from operator import add

# Initialize LLM
llm = get_open_ai_model()

# ---------------- STATE ---------------- #
class State(TypedDict):
    topic: str
    instagram: Annotated[str, add]
    linkedin: Annotated[str, add]
    twitter: Annotated[str, add]


# ---------------- NODES ---------------- #
def create_post_instagram(state: State) -> State:
    topic = state["topic"]
    response = llm.invoke(
        f"Create an Instagram post about {topic}. Keep it under 50 words."
    ).content
    return {"instagram": response}


def create_post_linkedin(state: State) -> State:
    topic = state["topic"]
    response = llm.invoke(
        f"Create a LinkedIn post about {topic}. Keep it under 200 words."
    ).content
    return {"linkedin": response}


def create_post_twitter(state: State) -> State:
    topic = state["topic"]
    response = llm.invoke(
        f"Create a Twitter post about {topic}. Keep it under 280 characters."
    ).content
    return {"twitter": response}


# ---------------- GRAPH ---------------- #
graph = StateGraph(State)

graph.add_node("Instagram_Node", create_post_instagram)
graph.add_node("LinkedIn_Node", create_post_linkedin)
graph.add_node("Twitter_Node", create_post_twitter)

# Parallel execution
graph.add_edge(START, "Instagram_Node")
graph.add_edge(START, "LinkedIn_Node")
graph.add_edge(START, "Twitter_Node")

graph.add_edge("Instagram_Node", END)
graph.add_edge("LinkedIn_Node", END)
graph.add_edge("Twitter_Node", END)

parallel_graph = graph.compile()


# ---------------- EXECUTION ---------------- #
result = parallel_graph.invoke({
    "topic": "The impact of AI on society"
})


# ---------------- PRINT FUNCTION ---------------- #
def print_posts(result):
    print("\n" + "=" * 60)
    print("🚀 GENERATED SOCIAL MEDIA POSTS")
    print("=" * 60)

    platforms = {
        "📸 Instagram": result.get("instagram", ""),
        "💼 LinkedIn": result.get("linkedin", ""),
        "🐦 Twitter": result.get("twitter", "")
    }

    for platform, content in platforms.items():
        print(f"\n{platform}")
        print("-" * len(platform))
        print(content)
        print("\n" + "-" * 60)


# Print output
print_posts(result)