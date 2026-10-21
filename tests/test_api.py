import json

import pytest
from fastapi.testclient import TestClient

import main
from src.generation.groq_client import LLMServiceUnavailableError, ModelUnavailableError


@pytest.fixture
def client(monkeypatch, tmp_path, sample_chunks):
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        "\n".join(json.dumps(chunk, ensure_ascii=False) for chunk in sample_chunks),
        encoding="utf-8",
    )
    monkeypatch.setenv("CHUNKS_PATH", str(chunks_path))
    monkeypatch.setenv("GROQ_GENERATION_MODEL", "test/model")

    logged = []
    monkeypatch.setattr(main, "init_db", lambda: None)
    monkeypatch.setattr(
        main, "log_conversation", lambda *args, **kwargs: logged.append(kwargs) or len(logged)
    )
    monkeypatch.setattr(main, "log_feedback", lambda *args, **kwargs: None)

    with TestClient(main.app) as test_client:
        test_client.logged = logged
        yield test_client


def test_health_reports_the_configured_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "generation_model": "test/model"}


def test_ask_returns_answer_sources_and_model(client, monkeypatch):
    def fake_answer(question, chunks, model):
        return {"answer": "La tristeza bovina...", "sources": [{"document": "garrapatas.pdf", "page": 4}]}

    monkeypatch.setattr(main, "answer_question", fake_answer)
    response = client.post("/ask", json={"question": "¿Qué es la tristeza bovina?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("La tristeza")
    assert body["sources"] == [{"document": "garrapatas.pdf", "page": 4}]
    assert body["model"] == "test/model"
    assert client.logged[-1]["generation_model"] == "test/model"


def test_off_topic_question_does_not_call_the_llm(client, monkeypatch):
    def must_not_be_called(*args, **kwargs):
        raise AssertionError("the LLM should not be called without context")

    monkeypatch.setattr(main, "answer_question", must_not_be_called)
    response = client.post("/ask", json={"question": "capital de Francia"})
    assert response.status_code == 200
    assert response.json()["answer"] == main.NO_DOCUMENTS_MESSAGE


def test_blank_question_is_rejected(client):
    assert client.post("/ask", json={"question": "   "}).status_code == 400
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_retired_model_returns_503_with_instructions_instead_of_500(client, monkeypatch):
    def retired(*args, **kwargs):
        raise ModelUnavailableError("llama-3.3-70b-versatile", "model_decommissioned")

    monkeypatch.setattr(main, "answer_question", retired)
    response = client.post("/ask", json={"question": "¿Qué es la tristeza bovina?"})
    assert response.status_code == 503
    assert "GROQ_GENERATION_MODEL" in response.json()["detail"]


def test_temporary_provider_error_returns_503(client, monkeypatch):
    def down(*args, **kwargs):
        raise LLMServiceUnavailableError("rate limited")

    monkeypatch.setattr(main, "answer_question", down)
    response = client.post("/ask", json={"question": "¿Qué es la tristeza bovina?"})
    assert response.status_code == 503


def test_feedback_validates_the_rating(client):
    assert client.post("/feedback", json={"conversation_id": 1, "rating": "up"}).status_code == 200
    assert client.post("/feedback", json={"conversation_id": 1, "rating": "meh"}).status_code == 422
