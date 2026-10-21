"""Metrics for evaluating document retrieval results."""

from src.evaluation.stats import bootstrap_mean_interval, wilson_interval


def first_relevant_rank(expected_ids: set[str], result_ids: list[str]) -> int | None:
    """1-based position of the first expected chunk in the results, or None."""
    return next(
        (
            position
            for position, chunk_id in enumerate(result_ids, start=1)
            if chunk_id in expected_ids
        ),
        None,
    )


def evaluate_retrieval(
    ground_truth: list[dict[str, str | list[str]]],
    retrieved_ids: list[list[str]],
) -> dict[str, float | int | tuple[float, float]]:
    """Calculate Hit Rate and MRR (with 95% intervals) for retrieved chunk IDs.

    Hit Rate measures the share of questions with at least one expected chunk
    in the retrieved results. Mean Reciprocal Rank (MRR) also rewards finding
    an expected chunk closer to the first position.
    """
    if len(ground_truth) != len(retrieved_ids):
        raise ValueError("ground_truth and retrieved_ids must have the same length")
    if not ground_truth:
        raise ValueError("ground_truth must not be empty")

    reciprocal_ranks = []
    for item, result_ids in zip(ground_truth, retrieved_ids, strict=True):
        rank = first_relevant_rank(set(item["expected_chunk_ids"]), result_ids)
        reciprocal_ranks.append(1 / rank if rank is not None else 0.0)

    total = len(ground_truth)
    hits = sum(1 for value in reciprocal_ranks if value > 0)

    return {
        "questions": total,
        "hit_rate": hits / total,
        "hit_rate_ci": wilson_interval(hits, total),
        "mrr": sum(reciprocal_ranks) / total,
        "mrr_ci": bootstrap_mean_interval(reciprocal_ranks),
    }
