import pytest

from src.config import DEFAULT_GENERATION_MODEL, get_settings


def test_defaults_use_a_supported_model():
    settings = get_settings()
    assert settings.generation_model == DEFAULT_GENERATION_MODEL
    assert settings.generation_model != "llama-3.3-70b-versatile"  # retired 2026-08-16
    assert settings.judge_model == settings.generation_model
    assert settings.top_k == 5


def test_model_can_be_changed_from_the_environment(monkeypatch):
    monkeypatch.setenv("GROQ_GENERATION_MODEL", "some/new-model")
    monkeypatch.setenv("GROQ_JUDGE_MODEL", "some/judge-model")
    settings = get_settings()
    assert settings.generation_model == "some/new-model"
    assert settings.judge_model == "some/judge-model"


def test_judge_follows_generation_model_when_not_set(monkeypatch):
    monkeypatch.setenv("GROQ_GENERATION_MODEL", "some/new-model")
    assert get_settings().judge_model == "some/new-model"


@pytest.mark.parametrize("value", ["0", "-3"])
def test_top_k_must_be_positive(monkeypatch, value):
    monkeypatch.setenv("RETRIEVAL_TOP_K", value)
    with pytest.raises(ValueError, match="RETRIEVAL_TOP_K"):
        get_settings()


def test_invalid_number_gives_a_clear_error(monkeypatch):
    monkeypatch.setenv("GENERATION_MAX_TOKENS", "a lot")
    with pytest.raises(ValueError, match="GENERATION_MAX_TOKENS must be an integer"):
        get_settings()
