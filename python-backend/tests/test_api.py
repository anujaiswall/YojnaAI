import pytest
from fastapi.testclient import TestClient

from app import main


@pytest.fixture
def client():
    main.cache._data.clear()
    return TestClient(main.app)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_chat_returns_answer_sources_and_steps(client, fake_ai):
    fake_ai()
    body = client.post("/api/chat", json={"query": "Can a small farmer get PM-KISAN?"}).json()

    assert "6000" in body["answer"]
    assert body["sources"] == ["RevisedPM-KISANOperationalGuidelines(English).pdf"]
    assert body["search_query"].startswith("search 1")
    assert len(body["steps"]) == 4
    assert body["cached"] is False


def test_repeat_question_is_served_from_cache(client, fake_ai):
    fake = fake_ai()
    client.post("/api/chat", json={"query": "What is PM-KISAN?"})
    second = client.post("/api/chat", json={"query": "  what is   pm-kisan "}).json()

    assert second["cached"] is True
    assert len(fake.searches) == 1  # no second search or LLM call


@pytest.mark.parametrize("bad", [{}, {"query": ""}, {"query": "hi"}, {"query": "x" * 501}])
def test_rejects_invalid_queries(client, bad):
    assert client.post("/api/chat", json=bad).status_code == 422


def test_upstream_failure_returns_502(client, monkeypatch):
    def boom(question):
        raise RuntimeError("Gemini timeout")

    monkeypatch.setattr(main.graph, "run", boom)
    response = client.post("/api/chat", json={"query": "What is PMAY?"})
    assert response.status_code == 502
