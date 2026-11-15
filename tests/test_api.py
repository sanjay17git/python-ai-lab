
import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert "chunks_indexed" in data


def test_empty_query_returns_422(client):
    response = client.post(
        "/ask",
        json={"query": ""}
    )

    assert response.status_code == 422


def test_ask_returns_answer(client):
    response = client.post(
        "/ask",
        json={
            "query": "What is the minimum attendance requirement?",
            "k": 3
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "answer" in data
    assert "citations" in data
    assert "total_ms" in data
    assert data["total_ms"] > 0


def test_search_returns_chunks(client):
    response = client.post(
        "/search",
        json={
            "query": "mobile phone during examination",
            "k": 5
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data["results"]) > 0
    assert "chunk_id" in data["results"][0]
    assert "score" in data["results"][0]


def test_ask_refusal_on_out_of_scope(client):
    response = client.post(
        "/ask",
        json={
            "query": "What time does the library close?",
            "k": 3
        }
    )

    assert response.status_code == 200

    answer = response.json()["answer"].lower()

    refusal_phrases = [
        "cannot answer",
        "can't answer",
        "don't have enough",
        "not enough information",
        "context does not contain",
        "provided documents"
    ]

    assert any(
        phrase in answer
        for phrase in refusal_phrases
    ), f"Expected refusal, but received: {answer}"
