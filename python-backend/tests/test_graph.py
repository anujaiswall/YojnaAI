from app import graph
from app.config import NOT_FOUND_ANSWER


def test_answers_when_first_search_is_relevant(fake_ai):
    fake = fake_ai(relevant_on_attempt=1)
    state = graph.run("Can a small farmer get PM-KISAN?")

    assert "6000" in state["answer"]
    assert len(fake.searches) == 1
    assert fake.answers == 1


def test_rewrites_and_searches_again_when_chunks_are_irrelevant(fake_ai):
    fake = fake_ai(relevant_on_attempt=2)
    state = graph.run("kisan money?")

    assert len(fake.searches) == 2
    # The second rewrite is told which query failed.
    assert fake.rewrites[1][1] == "search 1: kisan money?"
    assert fake.answers == 1
    assert any("not relevant" in step for step in state["steps"])


def test_gives_up_after_retries_without_calling_the_answer_model(fake_ai):
    fake = fake_ai(relevant_on_attempt=99)
    state = graph.run("What is the capital of France?")

    assert state["answer"] == NOT_FOUND_ANSWER
    assert state["chunks"] == []
    assert fake.answers == 0
