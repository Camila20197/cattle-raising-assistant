import json
from pathlib import Path

from src.retrieval.keyword_search import search_chunks
from src.retrieval.vector_search import search_vector
from src.retrieval.hybrid_search import hybrid_search


CHUNKS_PATH = Path("data/processed/chunks.jsonl")
INDEXED_CHUNKS_PATH = Path("data/processed/indexed_chunks.jsonl")


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file and return a list of dictionaries."""
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def print_results(title: str, results: list[dict]) -> None:
    """Print search results in a readable format."""
    print(f"\n{'=' * 70}")
    print(title)
    print("=" * 70)

    for position, result in enumerate(results, start=1):
        print(f"\n#{position}")
        print(f"ID: {result['id']}")
        print(f"Documento: {result['document']}")
        print(f"Página: {result['page']}")
        print(f"Score: {result['score']}")
        print(f"Texto: {result['text'][:300]}...")


def main():
    query = "¿Qué efectos tuvo la sequía sobre la ganancia de peso de los bovinos?"

    chunks = load_jsonl(CHUNKS_PATH)
    indexed_chunks = load_jsonl(INDEXED_CHUNKS_PATH)

    keyword_results = search_chunks(
        query,
        chunks,
        top_k=5,
    )

    vector_results = search_vector(
        query,
        indexed_chunks,
        top_k=5,
    )

    hybrid_results = hybrid_search(
        query,
        chunks,
        indexed_chunks,
        top_k=5,
    )

    print(f"\nPREGUNTA: {query}")

    print_results("KEYWORD SEARCH", keyword_results)
    print_results("VECTOR SEARCH", vector_results)
    print_results("HYBRID SEARCH", hybrid_results)


if __name__ == "__main__":
    main()