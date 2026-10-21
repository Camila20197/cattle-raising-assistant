"""Evaluate keyword, vector and hybrid retrieval against the ground truth set.

Computes two standard retrieval metrics for each method, each with a 95%
confidence interval (the ground truth is small, so the uncertainty matters):

- Hit Rate@k: fraction of questions where an expected chunk appears
  anywhere among the top-k results (Wilson interval).
- MRR (Mean Reciprocal Rank): average of 1/rank of the first expected chunk
  (0 if it never appears in the top-k). Rewards finding the right chunk near
  the top, not just somewhere in the list (bootstrap interval).

Run with:
    uv run python -m src.evaluation.evaluate_retrieval            # all methods
    uv run python -m src.evaluation.evaluate_retrieval --methods keyword
Vector and hybrid search need GEMINI_API_KEY (query embeddings).
"""

import argparse
import json
import logging
import os
from collections.abc import Callable

from src.config import get_settings
from src.evaluation.retrieval import evaluate_retrieval
from src.evaluation.stats import format_interval
from src.jsonl import read_jsonl
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.keyword_search import build_bm25_index, search_chunks
from src.retrieval.vector_search import search_vector

logger = logging.getLogger(__name__)

ALL_METHODS = ("keyword", "vector", "hybrid")


def build_search_functions(
    methods: list[str], top_k: int
) -> dict[str, Callable[[str], list[dict]]]:
    """Return one ``question -> results`` function per requested method."""
    settings = get_settings()
    chunks = read_jsonl(settings.chunks_path)
    bm25 = build_bm25_index(chunks)
    functions: dict[str, Callable[[str], list[dict]]] = {
        "keyword": lambda query: search_chunks(query, chunks, top_k=top_k, bm25=bm25)
    }

    if {"vector", "hybrid"} & set(methods):
        indexed_chunks = read_jsonl(settings.indexed_chunks_path)
        functions["vector"] = lambda query: search_vector(query, indexed_chunks, top_k=top_k)
        functions["hybrid"] = lambda query: hybrid_search(
            query, chunks, indexed_chunks, top_k=top_k
        )

    return {method: functions[method] for method in methods}


def print_report(results: dict[str, dict], top_k: int) -> None:
    """Print hit rate and MRR, with 95% intervals, for each method."""
    questions = next(iter(results.values()))["questions"]
    print(f"\nRetrieval evaluation — {questions} questions, top_k={top_k}")
    print(f"{'Method':<10}{'Hit Rate@' + str(top_k):<12}{'95% CI':<14}{'MRR':<8}{'95% CI':<14}")
    print("-" * 58)
    for method, metrics in results.items():
        print(
            f"{method:<10}{metrics['hit_rate']:<12.1%}"
            f"{format_interval(metrics['hit_rate_ci']):<14}"
            f"{metrics['mrr']:<8.3f}"
            f"{format_interval(metrics['mrr_ci'], as_percent=False):<14}"
        )

    best_method = max(results, key=lambda method: results[method]["mrr"])
    print(f"\nBest method by MRR: {best_method}")
    print(
        "Note: if two methods' intervals overlap heavily, this sample is too "
        "small to say that one is truly better."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=None)
    args = parser.parse_args()

    settings = get_settings()
    methods = args.methods or list(ALL_METHODS)
    if not os.getenv("GEMINI_API_KEY") and args.methods is None:
        logger.warning("GEMINI_API_KEY is not set: evaluating keyword search only.")
        methods = ["keyword"]

    with settings.ground_truth_path.open("r", encoding="utf-8") as file:
        ground_truth = json.load(file)

    results = {}
    for method, search_fn in build_search_functions(methods, settings.top_k).items():
        logger.info("Evaluating %s search on %s questions...", method, len(ground_truth))
        retrieved_ids = [
            [str(result["id"]) for result in search_fn(item["question"])]
            for item in ground_truth
        ]
        results[method] = evaluate_retrieval(ground_truth, retrieved_ids)

    print_report(results, settings.top_k)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
