"""Utilities for splitting extracted PDF pages into retrieval chunks."""


def split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks, preferring whitespace boundaries.

    Overlap keeps some context shared between adjacent chunks. This reduces the
    chance that information near a chunk boundary becomes difficult to retrieve.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be greater than or equal to zero and smaller than chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            whitespace_index = text.rfind(" ", start, end)
            if whitespace_index > start:
                end = whitespace_index

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def chunk_pages(
    pages: list[dict[str, str | int]], chunk_size: int = 1000, overlap: int = 200
) -> list[dict[str, str | int]]:
    """Split page records into chunks while preserving their source metadata."""
    chunks = []

    for page in pages:
        page_chunks = split_text(str(page["text"]), chunk_size, overlap)

        for chunk_index, text in enumerate(page_chunks, start=1):
            chunks.append(
                {
                    "id": f'{page["document"]}:{page["page"]}:{chunk_index}',
                    "document": page["document"],
                    "page": page["page"],
                    "chunk_index": chunk_index,
                    "text": text,
                }
            )

    return chunks
