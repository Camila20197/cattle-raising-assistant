"""End-to-end ingestion pipeline: raw PDFs -> chunks.jsonl -> indexed_chunks.jsonl.

This is the script Kestra automates (see the ingestion flow). It ties together
pieces that used to be run by hand, cell by cell, in a notebook:

    load_pdfs (pdf_loader.py) -> chunk_pages (chunker.py) -> index_chunks (vector_search.py)

Run manually with:
    uv run python -m src.ingestion.run_ingestion
"""

from pathlib import Path

from src.ingestion.chunker import chunk_pages
from src.ingestion.pdf_loader import load_pdfs, save_records
from src.retrieval.vector_search import index_chunks

RAW_PDF_DIR = Path("data/raw")
PAGES_PATH = Path("data/processed/pages.jsonl")
CHUNKS_PATH = Path("data/processed/chunks.jsonl")
INDEXED_CHUNKS_PATH = Path("data/processed/indexed_chunks.jsonl")


def main():
    print(f"Cargando PDFs desde {RAW_PDF_DIR}...")
    pages = load_pdfs(RAW_PDF_DIR)
    save_records(pages, PAGES_PATH)
    print(f"  {len(pages)} páginas extraídas -> {PAGES_PATH}")

    print("Dividiendo en chunks...")
    chunks = chunk_pages(pages)
    save_records(chunks, CHUNKS_PATH)
    print(f"  {len(chunks)} chunks creados -> {CHUNKS_PATH}")

    print("Generando embeddings (llama a la API de Gemini)...")
    indexed_chunks = index_chunks(chunks)
    save_records(indexed_chunks, INDEXED_CHUNKS_PATH)
    print(f"  {len(indexed_chunks)} chunks indexados -> {INDEXED_CHUNKS_PATH}")

    print("Ingesta completa.")


if __name__ == "__main__":
    main()