"""Vector search in Pinecone."""
from functools import lru_cache

from pydantic import BaseModel

from app.config import PINECONE_API_KEY, PINECONE_INDEX, TOP_K
from app.embeddings import embed_one


class Chunk(BaseModel):
    text: str
    source: str
    score: float


@lru_cache(maxsize=1)
def _index():
    from pinecone import Pinecone

    return Pinecone(api_key=PINECONE_API_KEY).Index(PINECONE_INDEX)


def search(query: str, top_k: int = TOP_K) -> list[Chunk]:
    """Embed the query and return the top_k most similar chunks."""
    result = _index().query(vector=embed_one(query), top_k=top_k, include_metadata=True)
    chunks = []
    for match in result.matches or []:
        meta = match.metadata or {}
        chunks.append(Chunk(text=meta.get("text", ""), source=meta.get("source", "unknown"), score=match.score))
    return chunks
