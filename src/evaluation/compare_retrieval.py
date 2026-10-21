"""Show side by side what keyword, vector and hybrid search return for one question.

A quick manual inspection tool (not an automated test). Run with:
    uv run python -m src.evaluation.compare_retrieval "¿Tu pregunta?"
Vector and hybrid search need GEMINI_API_KEY.
"""

import sys

from src.config import get_settings
from src.jsonl import read_jsonl
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.keyword_search import search_chunks
from src.retrieval.vector_search import search_vector

DEFAULT_QUERY = "¿Qué efectos tuvo la sequía sobre la ganancia de peso de los bovinos?"


def print_results(title: str, results: list[dict]) -> None:
    """Print search results in a readable format."""
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
    for position, result in enumerate(results, start=1):
        print(f"\n#{position}")
        print(f"ID: {result['id']}")
        print(f"Document: {result['document']}")
        print(f"Page: {result['page']}")
        print(f"Score: {result['score']}")
        print(f"Text: {result['text'][:300]}...")


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY
    settings = get_settings()
    chunks = read_jsonl(settings.chunks_path)
    indexed_chunks = read_jsonl(settings.indexed_chunks_path)

    print(f"\nQUESTION: {query}")
    print_results("KEYWORD SEARCH", search_chunks(query, chunks, top_k=settings.top_k))
    print_results("VECTOR SEARCH", search_vector(query, indexed_chunks, top_k=settings.top_k))
    print_results(
        "HYBRID SEARCH", hybrid_search(query, chunks, indexed_chunks, top_k=settings.top_k)
    )


if __name__ == "__main__":
    main()
