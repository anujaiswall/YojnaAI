"""Test setup: fake Gemini and Pinecone so tests run offline, with no API keys."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import llm, retrieval  # noqa: E402
from app.retrieval import Chunk  # noqa: E402

PM_KISAN = Chunk(
    text="PM-KISAN provides income support of Rs 6000 per year to landholding farmer families.",
    source="RevisedPM-KISANOperationalGuidelines(English).pdf",
    score=0.82,
)


class FakeAI:
    """Records calls and returns scripted answers."""

    def __init__(self, relevant_on_attempt: int = 1):
        self.relevant_on_attempt = relevant_on_attempt
        self.rewrites: list[tuple[str, str | None]] = []
        self.searches: list[str] = []
        self.answers = 0

    def rewrite_query(self, question, previous_query=None):
        self.rewrites.append((question, previous_query))
        return f"search {len(self.rewrites)}: {question}"

    def search(self, query, top_k=5):
        self.searches.append(query)
        return [PM_KISAN]

    def grade_chunks(self, question, chunks):
        ok = len(self.searches) >= self.relevant_on_attempt
        return llm.Grade(relevant=ok, reason="matches" if ok else "off topic")

    def generate_answer(self, question, chunks):
        self.answers += 1
        return "Eligible landholding farmer families get Rs 6000 per year (source: PM-KISAN guidelines)."


@pytest.fixture
def fake_ai(monkeypatch):
    def install(relevant_on_attempt: int = 1) -> FakeAI:
        fake = FakeAI(relevant_on_attempt)
        monkeypatch.setattr(llm, "rewrite_query", fake.rewrite_query)
        monkeypatch.setattr(llm, "grade_chunks", fake.grade_chunks)
        monkeypatch.setattr(llm, "generate_answer", fake.generate_answer)
        monkeypatch.setattr(retrieval, "search", fake.search)
        return fake

    return install
