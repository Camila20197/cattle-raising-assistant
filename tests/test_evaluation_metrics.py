import pytest

from src.evaluation.retrieval import evaluate_retrieval
from src.evaluation.stats import bootstrap_mean_interval, wilson_interval


def test_hit_rate_and_mrr():
    ground_truth = [
        {"question": "q1", "expected_chunk_ids": ["a"]},
        {"question": "q2", "expected_chunk_ids": ["b"]},
        {"question": "q3", "expected_chunk_ids": ["z"]},
    ]
    retrieved = [["a", "x"], ["x", "b"], ["x", "y"]]
    metrics = evaluate_retrieval(ground_truth, retrieved)
    assert metrics["questions"] == 3
    assert metrics["hit_rate"] == pytest.approx(2 / 3)
    assert metrics["mrr"] == pytest.approx((1 + 0.5 + 0) / 3)
    low, high = metrics["hit_rate_ci"]
    assert low < metrics["hit_rate"] < high


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        evaluate_retrieval([{"question": "q", "expected_chunk_ids": ["a"]}], [])


def test_wilson_interval_is_wide_for_small_samples():
    low_small, high_small = wilson_interval(9, 10)
    low_big, high_big = wilson_interval(90, 100)
    assert (high_small - low_small) > (high_big - low_big)
    assert high_small < 1.0 or low_small > 0.0


def test_wilson_interval_for_a_perfect_score_is_not_100_percent_certain():
    low, high = wilson_interval(10, 10)
    assert high == pytest.approx(1.0)
    assert low < 0.75  # 10/10 is compatible with a much lower true rate


def test_bootstrap_interval_is_reproducible():
    values = [1.0, 0.5, 0.0, 1.0, 0.333]
    assert bootstrap_mean_interval(values) == bootstrap_mean_interval(values)
    low, high = bootstrap_mean_interval(values)
    assert low <= sum(values) / len(values) <= high
