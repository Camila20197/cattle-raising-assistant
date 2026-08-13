"""Evaluate keyword, vector and hybrid retrieval against a ground truth set.

Computes two standard retrieval metrics for each method:

- Hit Rate@k: fraction of questions where the expected chunk appears
  anywhere among the top-k results.
- MRR (Mean Reciprocal Rank): average of 1/rank of the expected chunk
  across all questions (0 if it never appears in the top-k). Rewards
  finding the right chunk near the top, not just somewhere in the list.
"""

import json
from pathlib import Path
from typing import Callable

from src.retrieval.keyword_search import search_chunks
from src.retrieval.vector_search import search_vector
from src.retrieval.hybrid_search import hybrid_search

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
INDEXED_CHUNKS_PATH = Path("data/processed/indexed_chunks.jsonl")
GROUND_TRUTH_PATH = Path("data/evaluation/retrieval_ground_truth.json")
TOP_K = 5


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file and return a list of dictionaries."""
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def load_ground_truth(path: Path) -> list[dict]:
    """Load the ground truth JSON file and return a list of dictionaries."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_method(
    search_fn: Callable[[str], list[dict]],
    ground_truth: list[dict],
) -> tuple[float, float]:
    """Return (hit_rate, mrr) for a single search function over the ground truth."""
    hits = 0
    reciprocal_ranks = []

    for item in ground_truth:
        results = search_fn(item["question"])
        result_ids = [str(result["id"]) for result in results]
        expected_ids = set(item["expected_chunk_ids"])

        rank = next(
            (position for position, result_id in enumerate(result_ids, start=1)
             if result_id in expected_ids),
            None,
        )

        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)

    total = len(ground_truth)
    hit_rate = hits / total
    mrr = sum(reciprocal_ranks) / total
    return hit_rate, mrr


def print_report(results: dict[str, tuple[float, float]]) -> None:
    """Print a report of hit rate and MRR for each method."""
    print(f"\n{'Método':<15}{'Hit Rate@' + str(TOP_K):<15}{'MRR':<10}")
    print("-" * 40)
    for method, (hit_rate, mrr) in results.items():
        print(f"{method:<15}{hit_rate:<15.2%}{mrr:<10.3f}")

    best_method = max(results, key=lambda method: results[method][1])
    print(f"\nMejor método según MRR: {best_method}")


def main():
    chunks = load_jsonl(CHUNKS_PATH)
    indexed_chunks = load_jsonl(INDEXED_CHUNKS_PATH)
    ground_truth = load_ground_truth(GROUND_TRUTH_PATH)

    methods = {
        "Keyword": lambda query: search_chunks(query, chunks, top_k=TOP_K),
        "Vector": lambda query: search_vector(query, indexed_chunks, top_k=TOP_K),
        "Hybrid": lambda query: hybrid_search(query, chunks, indexed_chunks, top_k=TOP_K),
    }

    results = {
        method_name: evaluate_method(search_fn, ground_truth)
        for method_name, search_fn in methods.items()
    }

    print_report(results)


if __name__ == "__main__":
    main()