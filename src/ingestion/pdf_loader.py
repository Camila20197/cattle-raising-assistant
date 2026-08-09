"""Utilities for extracting text and metadata from PDF documents."""

import json
import logging
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def load_pdfs(data_dir: Path) -> list[dict[str, str | int]]:
    """Load every PDF in ``data_dir`` and return one record per text page.

    Each record preserves the document name and page number so that a future
    answer can cite where its information came from. Pages with no extractable
    text are skipped because they cannot be retrieved by a text-based search.
    """
    if not data_dir.is_dir():
        raise FileNotFoundError(f"PDF directory does not exist: {data_dir}")

    pages = []
    skipped_pages = 0

    for pdf_path in sorted(data_dir.glob("*.pdf")):
        with fitz.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text", sort=True).strip()

                if not text:
                    skipped_pages += 1
                    continue

                pages.append(
                    {
                        "document": pdf_path.name,
                        "page": page_number,
                        "text": text,
                    }
                )

    logger.info("Loaded %s text pages and skipped %s empty pages.", len(pages), skipped_pages)

    return pages


def save_records(records: list[dict[str, str | int]], output_path: Path) -> None:
    """Save records as JSON Lines, writing one JSON object per line.

    JSON Lines is convenient for data pipelines because each line is an
    independent record and can be processed without loading the whole file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_records(input_path: Path) -> list[dict[str, str | int]]:
    """Load records stored in a JSON Lines file."""
    with input_path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]
