import pytest

from src.ingestion.chunker import chunk_pages, split_text


def test_short_text_is_a_single_chunk():
    assert split_text("hola mundo", chunk_size=100, overlap=10) == ["hola mundo"]


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"palabra{i}" for i in range(200))
    chunks = split_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)
    # Consecutive chunks share some text thanks to the overlap.
    assert chunks[0].split()[-1] in chunks[1]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known bug found by this test: the overlap start can land in the middle "
        "of a word (e.g. 'tres' -> 'res'). Fixing it changes chunk boundaries, so "
        "it must ship together with re-ingestion and a regenerated ground truth. "
        "See README > Roadmap."
    ),
)
def test_chunks_do_not_cut_words():
    text = "uno dos tres cuatro cinco seis siete ocho nueve diez"
    for chunk in split_text(text, chunk_size=15, overlap=3):
        for word in chunk.split():
            assert word in text.split()


@pytest.mark.parametrize("chunk_size, overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_parameters_are_rejected(chunk_size, overlap):
    with pytest.raises(ValueError):
        split_text("texto", chunk_size=chunk_size, overlap=overlap)


def test_chunk_pages_keeps_source_metadata():
    pages = [{"document": "doc.pdf", "page": 3, "text": "contenido de la página"}]
    [chunk] = chunk_pages(pages)
    assert chunk == {
        "id": "doc.pdf:3:1",
        "document": "doc.pdf",
        "page": 3,
        "chunk_index": 1,
        "text": "contenido de la página",
    }


def test_ligatures_are_normalized_for_search():
    from src.ingestion.pdf_loader import clean_text

    assert clean_text("  deﬁciencia de seleﬂ ") == "deficiencia de selefl"
    assert clean_text("sin cambios: ñandú, 2,5%") == "sin cambios: ñandú, 2,5%"
