"""Hybrid retrieval that combines keyword and vector search rankings."""

from src.retrieval.keyword_search import search_chunks
from src.retrieval.vector_search import search_vector


def reciprocal_rank_fusion(
    result_lists: list[list[dict[str, str | int | float]]], rrf_k: int = 60
) -> list[dict[str, str | int | float]]:
    """Combine ranked lists using Reciprocal Rank Fusion (RRF).

    RRF adds ``1 / (rrf_k + rank)`` for each occurrence of a result. It uses
    result positions instead of combining incompatible raw scores.
    """
    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than zero")

    fused_scores = {}
    results_by_id = {}

    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            result_id = str(result["id"])
            fused_scores[result_id] = fused_scores.get(result_id, 0.0) + 1 / (rrf_k + rank)
            results_by_id[result_id] = result

    return [
        {**results_by_id[result_id], "score": score}
        for result_id, score in sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    ]


def hybrid_search(
    query: str,
    chunks: list[dict[str, str | int]],
    indexed_chunks: list[dict[str, str | int | list[float]]],
    top_k: int = 5,
    candidate_k: int = 20,
) -> list[dict[str, str | int | float]]:
    """Retrieve chunks with keyword and vector search, then fuse the rankings."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if candidate_k < top_k:
        raise ValueError("candidate_k must be greater than or equal to top_k")

    keyword_results = search_chunks(query, chunks, top_k=candidate_k)
    vector_results = search_vector(query, indexed_chunks, top_k=candidate_k)

    return reciprocal_rank_fusion([keyword_results, vector_results])[:top_k]
