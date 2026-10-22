"""Monitoring storage: conversations and user feedback in Postgres.

Monitoring is optional. When DATABASE_URL is not set (e.g. a lightweight
Cloud Run deploy without a database), every function here becomes a no-op
and the assistant keeps answering questions normally.
"""

import json
import logging
import os

import psycopg

logger = logging.getLogger(__name__)


def monitoring_enabled() -> bool:
    """True when a database is configured for conversations and feedback."""
    return bool(os.getenv("DATABASE_URL"))


def get_connection() -> psycopg.Connection:
    """Open a new connection using DATABASE_URL from the environment."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise EnvironmentError("DATABASE_URL is not set")

    return psycopg.connect(database_url)


def init_db() -> None:
    """Create the monitoring tables if they don't exist yet. Safe to call on every startup."""
    if not monitoring_enabled():
        logger.warning("DATABASE_URL is not set: monitoring (conversations and feedback) is disabled.")
        return
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources JSONB NOT NULL,
                retrieval_method TEXT NOT NULL,
                prompt_variant TEXT NOT NULL
            )
            """
        )
        # Added after the first release: ALTER keeps existing databases working.
        conn.execute(
            "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS generation_model TEXT"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id SERIAL PRIMARY KEY,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id),
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
                comment TEXT
            )
            """
        )
        conn.commit()


def log_conversation(
    question: str,
    answer: str,
    sources: list[dict[str, str | int]],
    retrieval_method: str = "bm25",
    prompt_variant: str = "A",
    generation_model: str | None = None,
) -> int | None:
    """Store a question/answer exchange and return its new conversation id.

    Returns None when monitoring is disabled.
    """
    if not monitoring_enabled():
        return None
    with get_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO conversations
                (question, answer, sources, retrieval_method, prompt_variant, generation_model)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                question,
                answer,
                json.dumps(sources),
                retrieval_method,
                prompt_variant,
                generation_model,
            ),
        ).fetchone()
        conn.commit()
        return row[0]


def log_feedback(conversation_id: int, rating: str, comment: str | None = None) -> None:
    """Store a thumbs up/down (and optional comment) linked to a conversation."""
    if not monitoring_enabled():
        return
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO feedback (conversation_id, rating, comment) VALUES (%s, %s, %s)",
            (conversation_id, rating, comment),
        )
        conn.commit()