from types import SimpleNamespace

import groq
import httpx
import pytest

from src.generation import rag
from src.generation.groq_client import (
    Completion,
    LLMError,
    LLMServiceUnavailableError,
    ModelUnavailableError,
    complete,
)
from src.generation.judge import parse_judgment


class FakeGroq:
    """Stands in for groq.Groq: records the request and returns a canned reply."""

    def __init__(self, content="respuesta", finish_reason="stop", error=None):
        self.requests = []
        self._content = content
        self._finish_reason = finish_reason
        self._error = error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.requests.append(request)
        if self._error:
            raise self._error
        message = SimpleNamespace(content=self._content)
        choice = SimpleNamespace(message=message, finish_reason=self._finish_reason)
        return SimpleNamespace(choices=[choice])


def make_status_error(error_class, status, body):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return error_class("provider error", response=response, body=body)


def test_complete_sends_the_requested_model():
    client = FakeGroq(content="  hola  ")
    completion = complete("prompt", model="some/model", temperature=0.1, client=client)
    assert completion == Completion(text="hola", finish_reason="stop", model="some/model")
    assert client.requests[0]["model"] == "some/model"
    assert "response_format" not in client.requests[0]


def test_complete_json_mode():
    client = FakeGroq(content="{}")
    complete("prompt", model="m", temperature=0, json_mode=True, client=client)
    assert client.requests[0]["response_format"] == {"type": "json_object"}


def test_decommissioned_model_raises_a_helpful_error():
    error = make_status_error(
        groq.BadRequestError,
        400,
        {"error": {"code": "model_decommissioned", "message": "decommissioned"}},
    )
    with pytest.raises(ModelUnavailableError) as raised:
        complete("p", model="llama-3.3-70b-versatile", temperature=0, client=FakeGroq(error=error))
    message = str(raised.value)
    assert "llama-3.3-70b-versatile" in message
    assert "GROQ_GENERATION_MODEL" in message
    assert "console.groq.com/docs/deprecations" in message


def test_unknown_model_is_also_reported_as_unavailable():
    error = make_status_error(
        groq.NotFoundError, 404, {"error": {"code": "model_not_found", "message": "no model"}}
    )
    with pytest.raises(ModelUnavailableError):
        complete("p", model="typo-model", temperature=0, client=FakeGroq(error=error))


def test_other_bad_requests_are_generic_llm_errors():
    error = make_status_error(groq.BadRequestError, 400, {"error": {"code": "context_length"}})
    with pytest.raises(LLMError) as raised:
        complete("p", model="m", temperature=0, client=FakeGroq(error=error))
    assert not isinstance(raised.value, ModelUnavailableError)


def test_rate_limit_is_a_temporary_error():
    error = make_status_error(groq.RateLimitError, 429, {"error": {"code": "rate_limit"}})
    with pytest.raises(LLMServiceUnavailableError):
        complete("p", model="m", temperature=0, client=FakeGroq(error=error))


@pytest.fixture
def fake_complete(monkeypatch):
    calls = []

    def _fake(prompt, **kwargs):
        calls.append({"prompt": prompt, **kwargs})
        return _fake.reply

    _fake.reply = Completion(text="La sequía redujo el peso [sequia.pdf: 1].", finish_reason="stop", model="m")
    _fake.calls = calls
    monkeypatch.setattr(rag, "complete", _fake)
    return _fake


def test_answer_uses_the_configured_model(monkeypatch, fake_complete, sample_chunks):
    monkeypatch.setenv("GROQ_GENERATION_MODEL", "configured/model")
    result = rag.answer_question("¿Qué pasó con la sequía?", sample_chunks[:1])
    assert fake_complete.calls[0]["model"] == "configured/model"
    assert result["answer"].startswith("La sequía")
    assert result["sources"] == [{"document": "sequia.pdf", "page": 1}]


def test_truncated_answer_is_not_returned_as_complete(fake_complete, sample_chunks):
    fake_complete.reply = Completion(text="La sequía redu", finish_reason="length", model="m")
    result = rag.answer_question("pregunta", sample_chunks[:1])
    assert result["answer"] == rag.TRUNCATED_MESSAGE


def test_empty_answer_gets_a_friendly_message(fake_complete, sample_chunks):
    fake_complete.reply = Completion(text="", finish_reason="stop", model="m")
    assert rag.answer_question("pregunta", sample_chunks[:1])["answer"] == rag.NO_ANSWER_MESSAGE


def test_answer_requires_context():
    with pytest.raises(ValueError):
        rag.answer_question("pregunta", [])


def test_sources_are_deduplicated(sample_chunks):
    sources = rag.extract_sources(sample_chunks)
    assert sources == [
        {"document": "sequia.pdf", "page": 1},
        {"document": "garrapatas.pdf", "page": 4},
    ]


def test_prompt_contains_safety_rules_and_context(sample_chunks):
    prompt = rag.build_prompt("¿Qué es la tristeza bovina?", sample_chunks)
    assert "Do not diagnose" in prompt
    assert "garrapatas.pdf" in prompt
    assert "¿Qué es la tristeza bovina?" in prompt


def test_parse_judgment_accepts_a_valid_reply():
    judgment = parse_judgment('{"relevance": "RELEVANT", "grounded": true, "explanation": "ok"}')
    assert judgment == {"relevance": "RELEVANT", "grounded": True, "explanation": "ok"}


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        '{"relevance": "MAYBE", "grounded": true}',
        '{"relevance": "RELEVANT", "grounded": "yes"}',
    ],
)
def test_parse_judgment_rejects_malformed_replies(raw):
    with pytest.raises(LLMError):
        parse_judgment(raw)
