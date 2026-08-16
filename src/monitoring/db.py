import json
import os

import psycopg


def get_connection() -> psycopg.Connection:
    """Open a new connection using DATABASE_URL from the environment."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise EnvironmentError("DATABASE_URL is not set")

    return psycopg.connect(database_url)


def init_db() -> None:
    """Create the monitoring tables if they don't exist yet. Safe to call on every startup."""
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
) -> int:
    """Store a question/answer exchange and return its new conversation id."""
    with get_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO conversations (question, answer, sources, retrieval_method, prompt_variant)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (question, answer, json.dumps(sources), retrieval_method, prompt_variant),
        ).fetchone()
        conn.commit()
        return row[0]


def log_feedback(conversation_id: int, rating: str, comment: str | None = None) -> None:
    """Store a thumbs up/down (and optional comment) linked to a conversation."""
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO feedback (conversation_id, rating, comment) VALUES (%s, %s, %s)",
            (conversation_id, rating, comment),
        )
        conn.commit()