"""All calls to Gemini: rewrite the query, grade retrieved chunks, write the answer."""
from functools import lru_cache

from pydantic import BaseModel, Field

from app.config import GEMINI_API_KEY, GEMINI_MODEL
from app.retrieval import Chunk


@lru_cache(maxsize=1)
def _client():
    from google import genai

    return genai.Client(api_key=GEMINI_API_KEY)


def _generate(prompt: str, schema: type[BaseModel] | None = None):
    from google.genai import types

    config = None
    if schema is not None:
        # Structured output: Gemini must return JSON matching the Pydantic model.
        config = types.GenerateContentConfig(response_mime_type="application/json", response_schema=schema)
    return _client().models.generate_content(model=GEMINI_MODEL, contents=prompt, config=config)


# ---------- 1. Query rewriting ----------
REWRITE_PROMPT = """You turn a citizen's question about Indian government schemes into a short,
clear search query for a document search engine.
- Keep scheme names, states, occupations, income and age details.
- Expand abbreviations if obvious (e.g. "PMAY" -> "Pradhan Mantri Awas Yojana").
- Remove filler words. Reply with the search query only.
{previous}
Question: {question}"""


def rewrite_query(question: str, previous_query: str | None = None) -> str:
    previous = ""
    if previous_query:
        previous = (
            f'The search query "{previous_query}" did not find relevant documents. '
            "Write a different query that uses other likely keywords."
        )
    text = _generate(REWRITE_PROMPT.format(question=question, previous=previous)).text or ""
    return text.strip().strip('"') or question


# ---------- 2. Relevance grading (Self-RAG check) ----------
class Grade(BaseModel):
    relevant: bool = Field(description="True if the chunks contain information that answers the question")
    reason: str = Field(description="One short sentence explaining the decision")


GRADE_PROMPT = """Do these document chunks contain information that answers the question?
Answer only about relevance, not about how good the answer would be.

Question: {question}

Chunks:
{chunks}"""


def grade_chunks(question: str, chunks: list[Chunk]) -> Grade:
    if not chunks:
        return Grade(relevant=False, reason="No chunks were retrieved.")
    joined = "\n\n".join(f"[{i + 1}] ({c.source}) {c.text[:1500]}" for i, c in enumerate(chunks))
    response = _generate(GRADE_PROMPT.format(question=question, chunks=joined), schema=Grade)
    if response.parsed is not None:
        return response.parsed
    return Grade.model_validate_json(response.text)


# ---------- 3. Grounded answer (same rules as the Node.js version) ----------
ANSWER_PROMPT = """You are SchemeWise, an AI assistant that helps users understand Indian Government Schemes.

STRICT RULES:
- ONLY answer from the supplied context.
- If the answer is not present, reply:
"I don't have information about this in my knowledge base. Please visit the official government website."
- Never hallucinate.
- Mention the source.
- Keep the answer simple.

CONTEXT:
{context}

QUESTION:
{question}"""


def generate_answer(question: str, chunks: list[Chunk]) -> str:
    context = "\n\n".join(f"[Source {i + 1}: {c.source}]\n{c.text}" for i, c in enumerate(chunks))
    return (_generate(ANSWER_PROMPT.format(context=context, question=question)).text or "").strip()
