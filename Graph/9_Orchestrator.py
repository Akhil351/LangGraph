from llm import get_open_ai_model
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from typing import Annotated, TypedDict, Literal, List
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor
from operator import add

# ---------------- SCHEMA ---------------- #
class LLMSchema(BaseModel):
    tasks:  List[str]=Field(description="List of tasks to perform")

# ---------------- LLMs ---------------- #
router_llm = get_open_ai_model().with_structured_output(LLMSchema)
content_llm = get_open_ai_model()

# ---------------- STATE ---------------- #
class State(TypedDict):
    tasks:  List[str]
    query: str
    result: Annotated[List[str], add]
    summary: str


# ---------------- NODES ---------------- #
def orchestrator_node(state: State) -> State:
    user_input = state["query"]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an orchestrator that breaks down a user query into tasks for the worker"),
            ("human", "User query: {user_input}. Please generate one prompt per task for the worker to complete")
        ]
    )

    chain= prompt | router_llm

    response=chain.invoke({"user_input":user_input})

    return {
        "tasks": response.tasks
    }

def execute(query: str):
    response=content_llm.invoke(f"Please answer the following query: {query}")
    return response.content


def worker_node(state: State)->State:
    tasks=state["tasks"]
    result=[]

    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        result_futures= executor.map(execute,tasks)
        for res in result_futures:
            result.append(res)


    return {
        "result": result
    }


def collector_node(state: State)-> State:
    results=state["result"]

    prompt=ChatPromptTemplate.from_messages(
        [
            ("system","You are a helpful assistant that summarizes the following results: {results}"),
            ("human","Please provide a concise summary of the above results.")
        ]
    )

    chain= prompt | content_llm

    summary=chain.invoke({"results":results}).content

    return {
        "summary": summary
    }




# ---------------- GRAPH ---------------- #
graph = StateGraph(State)

graph.add_node("Orchestrator_Node", orchestrator_node)
graph.add_node("Worker_Node", worker_node)
graph.add_node("Collector_Node", collector_node)

graph.add_edge(START, "Orchestrator_Node")
graph.add_edge("Orchestrator_Node", "Worker_Node")
graph.add_edge("Worker_Node", "Collector_Node")
graph.add_edge("Collector_Node", END)

graph=graph.compile()


result=graph.invoke({"query":"What is the capital of india , who is the current prime minister? and what is the population of India? in 2026 "})


print("Summary of results:",result["result"])
print("--"*60)
print(result["summary"])