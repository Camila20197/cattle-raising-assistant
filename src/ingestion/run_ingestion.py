"""End-to-end ingestion pipeline: raw PDFs -> chunks.jsonl -> indexed_chunks.jsonl.

This is the script Kestra automates (see the ingestion flow). It ties together
pieces that used to be run by hand, cell by cell, in a notebook:

    load_pdfs (pdf_loader.py) -> chunk_pages (chunker.py) -> index_chunks (vector_search.py)

Run manually with:
    uv run python -m src.ingestion.run_ingestion
"""

import logging
from pathlib import Path

from src.config import get_settings
from src.ingestion.chunker import chunk_pages
from src.ingestion.pdf_loader import load_pdfs
from src.jsonl import write_jsonl
from src.retrieval.vector_search import index_chunks

RAW_PDF_DIR = Path("data/raw")
PAGES_PATH = Path("data/processed/pages.jsonl")

logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()

    logger.info("Loading PDFs from %s...", RAW_PDF_DIR)
    pages = load_pdfs(RAW_PDF_DIR)
    write_jsonl(pages, PAGES_PATH)
    logger.info("%s pages extracted -> %s", len(pages), PAGES_PATH)

    logger.info("Splitting pages into chunks...")
    chunks = chunk_pages(pages)
    write_jsonl(chunks, settings.chunks_path)
    logger.info("%s chunks created -> %s", len(chunks), settings.chunks_path)

    logger.info("Generating embeddings (calls the Gemini API)...")
    indexed_chunks = index_chunks(chunks)
    write_jsonl(indexed_chunks, settings.indexed_chunks_path)
    logger.info("%s chunks indexed -> %s", len(indexed_chunks), settings.indexed_chunks_path)

    logger.info("Ingestion complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
