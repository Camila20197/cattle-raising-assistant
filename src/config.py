"""Runtime settings, read from environment variables (and the local ``.env``).

This module is the single place that knows *where* a setting comes from.
Every other module asks ``get_settings()`` for a value instead of hardcoding
it, so swapping a deprecated model, a path or ``top_k`` is a one-line change
in ``.env`` and never a code change (information hiding: the volatile
decision lives in one module only).
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Groq retired llama-3.3-70b-versatile on 2026-08-16 and recommends
# openai/gpt-oss-120b as its replacement:
# https://console.groq.com/docs/deprecations
DEFAULT_GENERATION_MODEL = "openai/gpt-oss-120b"


@dataclass(frozen=True)
class Settings:
    generation_model: str
    judge_model: str
    generation_temperature: float
    generation_max_tokens: int
    top_k: int
    chunks_path: Path
    indexed_chunks_path: Path
    ground_truth_path: Path
    llm_results_path: Path


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from error


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a number, got {raw!r}") from error


def get_settings() -> Settings:
    """Build the current settings from the environment.

    Read on every call (it is cheap) so tests and long-running processes
    always see the current environment instead of a stale cached copy.
    """
    load_dotenv()

    generation_model = os.getenv("GROQ_GENERATION_MODEL") or DEFAULT_GENERATION_MODEL
    # The judge defaults to the generation model, but can be set separately
    # (e.g. to use a stronger or cheaper model only for evaluation).
    judge_model = os.getenv("GROQ_JUDGE_MODEL") or generation_model

    top_k = _get_int("RETRIEVAL_TOP_K", 5)
    if top_k <= 0:
        raise ValueError("RETRIEVAL_TOP_K must be greater than zero")

    return Settings(
        generation_model=generation_model,
        judge_model=judge_model,
        generation_temperature=_get_float("GENERATION_TEMPERATURE", 0.1),
        # Reasoning models such as gpt-oss spend part of this budget on their
        # internal reasoning, so it is kept generous to avoid truncated answers.
        generation_max_tokens=_get_int("GENERATION_MAX_TOKENS", 4_096),
        top_k=top_k,
        chunks_path=Path(os.getenv("CHUNKS_PATH", "data/processed/chunks.jsonl")),
        indexed_chunks_path=Path(
            os.getenv("INDEXED_CHUNKS_PATH", "data/processed/indexed_chunks.jsonl")
        ),
        ground_truth_path=Path(
            os.getenv("GROUND_TRUTH_PATH", "data/evaluation/retrieval_ground_truth.json")
        ),
        llm_results_path=Path(
            os.getenv("LLM_RESULTS_PATH", "data/evaluation/llm_eval_results.json")
        ),
    )
