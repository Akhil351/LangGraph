from llm import get_open_ai_model
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, Literal
from pydantic import BaseModel, Field

# ---------------- SCHEMA ---------------- #
class LLMSchema(BaseModel):
    funny_flag: Literal["funny", "not funny"] = Field(description="Whether the joke is funny or not")
    feedback: str = Field(description="Feedback on the joke")

# ---------------- LLMs ---------------- #
router_llm = get_open_ai_model().with_structured_output(LLMSchema)
content_llm = get_open_ai_model()

# ---------------- STATE ---------------- #
class State(TypedDict):
    topic: str
    joke: str
    funny_flag: str
    feedback: str
    max_iterations: int


# ---------------- NODES ---------------- #
def joke_generator_node(state: State) -> State:
    topic = state["topic"]

    if state.get("feedback"):
        response = content_llm.invoke(
            f"Please improve this joke based on feedback.\n"
            f"Joke: {state['joke']}\n"
            f"Feedback: {state['feedback']}"
        )
    else:
        response = content_llm.invoke(
            f"Create a funny joke about: {topic}"
        )

    return {
        "joke": response.content
    }


def evaluator_node(state: State) -> State:
    joke = state["joke"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a comedy critic. Evaluate the joke and suggest improvements."),
        ("human",
         "Evaluate this joke:\n{joke}\n\n"
         "Respond with 'funny' or 'not funny' and provide feedback if needed.")
    ])

    chain = prompt | router_llm

    response = chain.invoke({"joke": joke})

    return {
        "funny_flag": response.funny_flag,
        "feedback": response.feedback,
        "max_iterations": state["max_iterations"] + 1
    }


def check_iteration(state: State) -> str:
    if state.get("funny_flag") == "funny":
        return "end"
    elif state["max_iterations"] < 5:
        return "evaluate_node"
    else:
        return "end"


# ---------------- GRAPH ---------------- #
graph = StateGraph(State)

graph.add_node("Joke_Generator", joke_generator_node)
graph.add_node("Evaluator", evaluator_node)

graph.add_edge(START, "Joke_Generator")

graph.add_conditional_edges(
    "Joke_Generator",
    check_iteration,
    {
        "evaluate_node": "Evaluator",
        "end": END
    }
)

graph.add_edge("Evaluator", "Joke_Generator")

graph = graph.compile()


result = graph.invoke({
    "topic": "AI",
    "joke": "",
    "funny_flag": "",
    "feedback": "",
    "max_iterations": 0
})


print("\n=== FINAL RESULT ===")
print(f"Topic          : {result.get('topic')}")
print(f"Final Joke     : {result.get('joke')}")
print(f"Funny Flag     : {result.get('funny_flag')}")
print(f"Feedback       : {result.get('feedback')}")
print(f"Iterations Used: {result.get('max_iterations')}")
print("====================\n")