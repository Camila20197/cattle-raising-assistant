"""Metrics for evaluating document retrieval results."""


def evaluate_retrieval(
    ground_truth: list[dict[str, str | list[str]]],
    retrieved_ids: list[list[str]],
) -> dict[str, float]:
    """Calculate Hit Rate and MRR for retrieved chunk IDs.

    Hit Rate measures the share of questions with at least one expected chunk
    in the retrieved results. Mean Reciprocal Rank (MRR) also rewards finding
    an expected chunk closer to the first position.
    """
    if len(ground_truth) != len(retrieved_ids):
        raise ValueError("ground_truth and retrieved_ids must have the same length")
    if not ground_truth:
        raise ValueError("ground_truth must not be empty")

    hits = 0
    reciprocal_ranks = []

    for item, result_ids in zip(ground_truth, retrieved_ids, strict=True):
        expected_ids = set(item["expected_chunk_ids"])
        rank = next(
            (
                position
                for position, chunk_id in enumerate(result_ids, start=1)
                if chunk_id in expected_ids
            ),
            None,
        )

        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)

    return {
        "hit_rate": hits / len(ground_truth),
        "mrr": sum(reciprocal_ranks) / len(ground_truth),
    }
