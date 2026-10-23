"""Utilities for extracting text and metadata from PDF documents."""

import logging
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Typographic ligatures some PDFs use (e.g. "deﬁciencia" with a single "ﬁ"
# glyph). Left as is, a search for "deficiencia" would never match them.
LIGATURES = str.maketrans({"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"})


def clean_text(text: str) -> str:
    """Normalize extracted text so keyword search matches what people type."""
    return text.translate(LIGATURES).strip()


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
                text = clean_text(page.get_text("text", sort=True))

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
