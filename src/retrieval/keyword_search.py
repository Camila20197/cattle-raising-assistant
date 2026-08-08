"""A simple keyword-based retrieval baseline for document chunks."""

import re
from collections import Counter

TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)
STOP_WORDS = {
    "a",
    "al",
    "ante",
    "bajo",
    "con",
    "contra",
    "como",
    "cómo",
    "de",
    "del",
    "desde",
    "donde",
    "el",
    "en",
    "entre",
    "es",
    "esta",
    "este",
    "la",
    "las",
    "lo",
    "los",
    "para",
    "por",
    "que",
    "se",
    "sin",
    "su",
    "sus",
    "un",
    "una",
    "y",
}


def tokenize(text: str) -> list[str]:
    """Convert text into searchable tokens and remove common Spanish words."""
    tokens = TOKEN_PATTERN.findall(text.casefold())
    return [token for token in tokens if token not in STOP_WORDS]


def search_chunks(
    query: str, chunks: list[dict[str, str | int]], top_k: int = 5
) -> list[dict[str, str | int]]:
    """Return the ``top_k`` chunks with the highest keyword-match score.

    The score is the total number of occurrences of query words in a chunk.
    Chunks with no matching words are excluded from the results.
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    scored_chunks = []

    for chunk in chunks:
        text_tokens = Counter(tokenize(str(chunk["text"])))
        score = sum(text_tokens[token] for token in query_tokens)

        if score > 0:
            scored_chunks.append({**chunk, "score": score})

    return sorted(
        scored_chunks,
        key=lambda chunk: (-int(chunk["score"]), str(chunk["id"])),
    )[:top_k]
