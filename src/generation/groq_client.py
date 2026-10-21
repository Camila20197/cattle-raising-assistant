"""The only module that talks to the Groq API.

Generation (``rag.py``) and LLM-as-a-judge (``judge.py``) both go through
``complete()``. Keeping every provider detail here (SDK, error types, model
names) means a provider change or a model deprecation touches one file,
and the rest of the code depends on a small, stable function instead of on
the Groq SDK (low coupling / dependency inversion).

Embeddings stay on Gemini (``vector_search.py``): Groq has no embeddings
endpoint, and the Gemini quota problem only affected generation and judging.
"""

import os
from dataclasses import dataclass

import groq
from dotenv import load_dotenv

DEPRECATIONS_URL = "https://console.groq.com/docs/deprecations"
_MODEL_GONE_CODES = {"model_decommissioned", "model_not_found"}


class LLMError(RuntimeError):
    """Base error for any failure while calling the LLM provider."""


class ModelUnavailableError(LLMError):
    """The configured model was retired or does not exist on the provider."""

    def __init__(self, model: str, provider_message: str):
        self.model = model
        super().__init__(
            f"Groq model '{model}' is not available ({provider_message}). "
            f"It was probably deprecated: check {DEPRECATIONS_URL} and set "
            "GROQ_GENERATION_MODEL (and GROQ_JUDGE_MODEL) in your .env to a "
            "supported model."
        )


class LLMServiceUnavailableError(LLMError):
    """Temporary provider problem: rate limit, timeout or server error."""


@dataclass(frozen=True)
class Completion:
    text: str
    finish_reason: str | None
    model: str


def get_groq_client() -> groq.Groq:
    """Create a Groq client using ``GROQ_API_KEY`` from the environment."""
    load_dotenv()

    if not os.getenv("GROQ_API_KEY"):
        raise EnvironmentError("GROQ_API_KEY is not set")

    return groq.Groq()


def _error_code(error: groq.APIStatusError) -> str | None:
    """Extract the provider error code (e.g. ``model_decommissioned``), if any."""
    body = error.body
    if isinstance(body, dict):
        inner = body.get("error", body)
        if isinstance(inner, dict):
            code = inner.get("code")
            return str(code) if code else None
    return None


def _is_model_gone(error: groq.APIStatusError) -> bool:
    if _error_code(error) in _MODEL_GONE_CODES:
        return True
    message = str(error).lower()
    return "decommissioned" in message or (
        error.status_code == 404 and "model" in message
    )


def complete(
    prompt: str,
    *,
    model: str,
    temperature: float,
    max_tokens: int | None = None,
    json_mode: bool = False,
    client: groq.Groq | None = None,
) -> Completion:
    """Send one user prompt to Groq and return the text of the first choice.

    Provider exceptions are translated into this module's own error types,
    so callers never need to import the Groq SDK.
    """
    client = client or get_groq_client()
    request: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if max_tokens is not None:
        request["max_tokens"] = max_tokens
    if json_mode:
        request["response_format"] = {"type": "json_object"}

    try:
        response = client.chat.completions.create(**request)
    except (groq.BadRequestError, groq.NotFoundError) as error:
        if _is_model_gone(error):
            raise ModelUnavailableError(model, str(error)) from error
        raise LLMError(f"Groq rejected the request: {error}") from error
    except (groq.RateLimitError, groq.APIConnectionError, groq.InternalServerError) as error:
        raise LLMServiceUnavailableError(f"Groq is temporarily unavailable: {error}") from error

    choice = response.choices[0]
    return Completion(
        text=(choice.message.content or "").strip(),
        finish_reason=choice.finish_reason,
        model=model,
    )
