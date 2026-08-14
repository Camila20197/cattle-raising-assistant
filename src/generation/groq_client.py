"""Groq client factory, used for answer generation and LLM-as-a-judge.

Kept separate from Gemini on purpose: embeddings (vector_search.py) stay on
Gemini, since that quota was never the problem. Only the two pieces that hit
the free-tier daily wall -- generation and judging -- move to Groq.
"""

import os

from dotenv import load_dotenv
from groq import Groq

GENERATION_MODEL = "llama-3.3-70b-versatile"


def get_groq_client() -> Groq:
    """Create a Groq client using ``GROQ_API_KEY`` from the environment."""
    load_dotenv()

    if not os.getenv("GROQ_API_KEY"):
        raise EnvironmentError("GROQ_API_KEY is not set")

    return Groq()