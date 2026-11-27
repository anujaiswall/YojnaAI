"""The RAG flow as a LangGraph graph.

    rewrite -> retrieve -> grade --relevant--> answer
                 ^           |
                 |  not relevant, retries left
                 +-----------+
                             |  not relevant, no retries left
                             +--> not_found

The grade step is a Self-RAG check: before answering, the LLM checks that the
retrieved chunks actually answer the question. If not, the query is rewritten
with different keywords and the search runs again.
"""
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app import llm, retrieval
from app.config import MAX_RETRIES, NOT_FOUND_ANSWER
from app.retrieval import Chunk


class RAGState(TypedDict, total=False):
    question: str
    query: str
    chunks: list[Chunk]
    relevant: bool
    attempts: int
    answer: str
    steps: list[str]  # human-readable trace, shown to the user


def _log(state: RAGState, message: str) -> list[str]:
    return [*state.get("steps", []), message]


def rewrite_node(state: RAGState) -> RAGState:
    previous = state.get("query") if state.get("attempts") else None
    query = llm.rewrite_query(state["question"], previous_query=previous)
    return {"query": query, "steps": _log(state, f'Rewrote question as: "{query}"')}


def retrieve_node(state: RAGState) -> RAGState:
    chunks = retrieval.search(state["query"])
    sources = sorted({c.source for c in chunks})
    return {
        "chunks": chunks,
        "attempts": state.get("attempts", 0) + 1,
        "steps": _log(state, f"Retrieved {len(chunks)} chunks from {len(sources)} document(s)"),
    }


def grade_node(state: RAGState) -> RAGState:
    grade = llm.grade_chunks(state["question"], state.get("chunks", []))
    verdict = "relevant" if grade.relevant else "not relevant"
    return {"relevant": grade.relevant, "steps": _log(state, f"Checked chunks: {verdict} ({grade.reason})")}


def answer_node(state: RAGState) -> RAGState:
    answer = llm.generate_answer(state["question"], state["chunks"])
    return {"answer": answer, "steps": _log(state, "Wrote an answer from the retrieved chunks")}


def not_found_node(state: RAGState) -> RAGState:
    return {"answer": NOT_FOUND_ANSWER, "chunks": [], "steps": _log(state, "No relevant documents found; stopped")}


def route_after_grade(state: RAGState) -> str:
    if state.get("relevant"):
        return "answer"
    if state.get("attempts", 0) <= MAX_RETRIES:
        return "rewrite"
    return "not_found"


def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("answer", answer_node)
    graph.add_node("not_found", not_found_node)

    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade", route_after_grade, {"answer": "answer", "rewrite": "rewrite", "not_found": "not_found"}
    )
    graph.add_edge("answer", END)
    graph.add_edge("not_found", END)
    return graph.compile()


def run(question: str) -> RAGState:
    return build_graph().invoke({"question": question, "attempts": 0, "steps": []})
