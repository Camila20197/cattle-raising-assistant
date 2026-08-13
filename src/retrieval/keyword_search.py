"""A BM25-based keyword retrieval baseline for document chunks."""

import re

from rank_bm25 import BM25Okapi

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


def build_bm25_index(chunks: list[dict[str, str | int]]) -> BM25Okapi:
    """Build a BM25 index once, for reuse across many searches.

    A one-off script (like the evaluation scripts) can afford to rebuild
    this on every call, but a long-lived process like an API should build
    it once at startup and reuse it -- otherwise every request re-scans the
    whole corpus for no reason.
    """
    tokenized_corpus = [tokenize(str(chunk["text"])) for chunk in chunks]
    return BM25Okapi(tokenized_corpus)


def search_chunks(
    query: str,
    chunks: list[dict[str, str | int]],
    top_k: int = 5,
    bm25: BM25Okapi | None = None,
) -> list[dict[str, str | int | float]]:
    """Return the ``top_k`` chunks with the highest BM25 score for the query.

    Unlike raw term-frequency counting, BM25 downweights words that are
    common across the whole corpus (e.g. "bovinos" appearing in almost every
    chunk) and upweights words that are rare and therefore distinctive for
    a given chunk (e.g. "sequía"). It also saturates the contribution of a
    word that repeats many times within the same chunk, so a chunk cannot
    win purely by repeating one term.

    ``bm25`` lets a caller pass in an index built once with
    ``build_bm25_index`` instead of rebuilding it on every call. If omitted,
    one is built fresh from ``chunks`` (the previous, still-correct
    behaviour used by the evaluation scripts).
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    bm25 = bm25 or build_bm25_index(chunks)
    scores = bm25.get_scores(query_tokens)

    scored_chunks = [
        {**chunk, "score": float(score)}
        for chunk, score in zip(chunks, scores, strict=True)
        if score > 0
    ]

    return sorted(
        scored_chunks,
        key=lambda chunk: (-float(chunk["score"]), str(chunk["id"])),
    )[:top_k]