"""Local embeddings with sentence-transformers (no embedding API needed)."""
from functools import lru_cache

from app.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _model():
    # Imported here so tests and the API can start without loading torch.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    """Return one normalized 384-dim vector per text."""
    vectors = _model().encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vectors]


def embed_one(text: str) -> list[float]:
    return embed([text])[0]
