import pytest

from src.retrieval.hybrid_search import reciprocal_rank_fusion
from src.retrieval.keyword_search import build_bm25_index, search_chunks, tokenize


def test_tokenize_lowercases_and_removes_stop_words():
    assert tokenize("¿Cómo afecta LA Sequía a los bovinos?") == ["afecta", "sequía", "bovinos"]


def test_keyword_search_ranks_the_matching_chunk_first(sample_chunks):
    results = search_chunks("tristeza bovina", sample_chunks, top_k=2)
    assert results[0]["id"] == "garrapatas.pdf:4:2"
    assert results[0]["score"] > 0


def test_keyword_search_returns_nothing_for_unrelated_questions(sample_chunks):
    assert search_chunks("capital de Francia", sample_chunks, top_k=3) == []


def test_prebuilt_index_gives_the_same_results(sample_chunks):
    bm25 = build_bm25_index(sample_chunks)
    query = "garrapata común"
    assert search_chunks(query, sample_chunks, bm25=bm25) == search_chunks(query, sample_chunks)


def test_top_k_must_be_positive(sample_chunks):
    with pytest.raises(ValueError):
        search_chunks("sequía", sample_chunks, top_k=0)


def test_rrf_rewards_documents_found_by_both_methods():
    keyword = [{"id": "a"}, {"id": "b"}]
    vector = [{"id": "b"}, {"id": "c"}]
    fused = reciprocal_rank_fusion([keyword, vector])
    assert [item["id"] for item in fused][0] == "b"
    assert {item["id"] for item in fused} == {"a", "b", "c"}
