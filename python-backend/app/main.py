"""FastAPI server for Yojna-AI (Python version of server.js).

Run:  uvicorn app.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from app import graph
from app.cache import TTLCache
from app.config import CACHE_MAX_ITEMS, CACHE_TTL_SECONDS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("yojna-ai")

app = FastAPI(title="Yojna-AI", version="2.0.0", description="Answers questions about Indian government schemes from official documents.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

cache = TTLCache(max_items=CACHE_MAX_ITEMS, ttl_seconds=CACHE_TTL_SECONDS)


class ChatRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500, examples=["Can a small farmer in Karnataka get PM-KISAN?"])

    @field_validator("query")
    @classmethod
    def clean(cls, value: str) -> str:
        # Collapse whitespace and drop control characters before the text reaches the LLM.
        value = "".join(ch for ch in value if ch.isprintable())
        value = " ".join(value.split())
        if len(value) < 3:
            raise ValueError("query is too short")
        return value


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    search_query: str | None = None
    steps: list[str] = []
    cached: bool = False


@app.get("/health")
def health():
    return {"status": "ok", "cached_answers": len(cache)}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    hit = cache.get(request.query)
    if hit is not None:
        return hit.model_copy(update={"cached": True})

    try:
        state = graph.run(request.query)
    except Exception as exc:  # network or API errors from Gemini / Pinecone
        log.exception("chat failed")
        raise HTTPException(status_code=502, detail="Upstream AI service failed, please try again.") from exc

    sources = sorted({c.source for c in state.get("chunks", [])})
    response = ChatResponse(
        answer=state["answer"],
        sources=sources,
        search_query=state.get("query"),
        steps=state.get("steps", []),
    )
    cache.set(request.query, response)
    return response
