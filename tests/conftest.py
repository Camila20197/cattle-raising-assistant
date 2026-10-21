"""Shared fixtures. Tests never call Groq, Gemini or Postgres: every external
dependency is replaced by a fake, so the suite is fast, free and offline."""

import pytest

SETTINGS_ENV_VARS = [
    "GROQ_GENERATION_MODEL",
    "GROQ_JUDGE_MODEL",
    "GENERATION_TEMPERATURE",
    "GENERATION_MAX_TOKENS",
    "RETRIEVAL_TOP_K",
    "CHUNKS_PATH",
    "LLM_RESULTS_PATH",
]


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    """Ignore the developer's real .env so tests always start from defaults."""
    monkeypatch.setattr("src.config.load_dotenv", lambda: None)
    for name in SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def sample_chunks() -> list[dict]:
    return [
        {
            "id": "sequia.pdf:1:1",
            "document": "sequia.pdf",
            "page": 1,
            "text": "La sequía redujo la ganancia de peso de los terneros en Gualeguaychú.",
        },
        {
            "id": "garrapatas.pdf:4:1",
            "document": "garrapatas.pdf",
            "page": 4,
            "text": "Rhipicephalus microplus es la garrapata común del bovino.",
        },
        {
            "id": "garrapatas.pdf:4:2",
            "document": "garrapatas.pdf",
            "page": 4,
            "text": "La tristeza bovina es transmitida por la garrapata.",
        },
    ]
