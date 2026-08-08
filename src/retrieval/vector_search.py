"""Gemini embedding and in-memory vector search utilities."""

import math
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSIONS = 768
EMBEDDING_BATCH_SIZE = 49
EMBEDDINGS_PER_MINUTE = 98


def get_client() -> genai.Client:
    """Create a Gemini client using ``GEMINI_API_KEY`` from the environment."""
    load_dotenv()

    if not os.getenv("GEMINI_API_KEY"):
        raise EnvironmentError("GEMINI_API_KEY is not set")

    return genai.Client()


def normalize(vector: list[float]) -> list[float]:
    """Return a unit-length vector for cosine similarity calculations."""
    magnitude = math.sqrt(sum(value * value for value in vector))

    if magnitude == 0:
        raise ValueError("Cannot normalize a zero-length vector")

    return [value / magnitude for value in vector]


def embed_texts(texts: list[str], input_type: str) -> list[list[float]]:
    """Generate normalized embeddings for retrieval documents or questions."""
    if not texts:
        return []
    if input_type not in {"document", "query"}:
        raise ValueError("input_type must be either 'document' or 'query'")

    client = get_client()
    embeddings = []
    embeddings_in_window = 0

    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
        if embeddings_in_window + len(batch) > EMBEDDINGS_PER_MINUTE:
            time.sleep(60)
            embeddings_in_window = 0

        if input_type == "document":
            prepared_batch = [f"title: cattle health documents | text: {text}" for text in batch]
        else:
            prepared_batch = [f"task: question answering | query: {text}" for text in batch]

        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[
                types.Content(parts=[types.Part.from_text(text=text)])
                for text in prepared_batch
            ],
            config=types.EmbedContentConfig(
                output_dimensionality=EMBEDDING_DIMENSIONS,
            ),
        )
        embeddings.extend(normalize(list(embedding.values)) for embedding in response.embeddings)
        embeddings_in_window += len(batch)

    return embeddings


def index_chunks(chunks: list[dict[str, str | int]]) -> list[dict[str, str | int | list[float]]]:
    """Add document embeddings to chunks for in-memory vector retrieval."""
    embeddings = embed_texts([str(chunk["text"]) for chunk in chunks], input_type="document")

    return [
        {**chunk, "embedding": embedding}
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]


def search_vector(
    query: str,
    indexed_chunks: list[dict[str, str | int | list[float]]],
    top_k: int = 5,
) -> list[dict[str, str | int | float]]:
    """Return the chunks with the highest cosine similarity to a query."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    query_embedding = embed_texts([query], input_type="query")[0]
    scored_chunks = []

    for chunk in indexed_chunks:
        embedding = chunk["embedding"]
        if not isinstance(embedding, list):
            raise ValueError("Each indexed chunk must contain an embedding list")

        score = sum(
            query_value * document_value
            for query_value, document_value in zip(query_embedding, embedding, strict=True)
        )
        scored_chunks.append(
            {
                key: value
                for key, value in chunk.items()
                if key != "embedding"
            }
            | {"score": score}
        )

    return sorted(scored_chunks, key=lambda chunk: float(chunk["score"]), reverse=True)[:top_k]
