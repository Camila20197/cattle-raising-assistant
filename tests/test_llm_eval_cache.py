from src.evaluation import evaluate_llm
from src.evaluation.evaluate_llm import (
    LEGACY_MODEL,
    cache_key,
    get_run,
    load_results,
    migrate_legacy_results,
    summarize,
)


def test_cache_key_changes_with_the_model():
    key_old = cache_key("prompt_a", "llama-3.3-70b-versatile", "llama-3.3-70b-versatile")
    key_new = cache_key("prompt_a", "openai/gpt-oss-120b", "openai/gpt-oss-120b")
    assert key_old != key_new
    assert cache_key("prompt_a", "m", "judge-1") != cache_key("prompt_a", "m", "judge-2")


def test_legacy_results_are_kept_but_not_reused_for_a_new_model():
    legacy = {"prompt_a": {"q1": {"relevance": "RELEVANT", "grounded": True}}, "prompt_b": {}}
    results = migrate_legacy_results(legacy)

    old_run = get_run(results, "prompt_a", LEGACY_MODEL, LEGACY_MODEL)
    assert "q1" in old_run["judgments"]

    new_run = get_run(results, "prompt_a", "openai/gpt-oss-120b", "openai/gpt-oss-120b")
    assert new_run["judgments"] == {}


def test_load_results_migrates_old_files(tmp_path):
    path = tmp_path / "results.json"
    path.write_text('{"prompt_a": {"q": {"relevance": "RELEVANT", "grounded": true}}}', encoding="utf-8")
    results = load_results(path)
    assert results["format_version"] == evaluate_llm.RESULTS_FORMAT_VERSION
    assert len(results["runs"]) == 1


def test_evaluate_prompt_does_not_replay_judgments_from_another_model(
    monkeypatch, tmp_path, sample_chunks
):
    """Regression test for the reviewer's finding: after swapping the model,
    every question must be generated and judged again."""
    results = migrate_legacy_results(
        {"prompt_a": {"¿Qué es la tristeza bovina?": {"relevance": "NON_RELEVANT", "grounded": False}}}
    )
    generated_with = []

    def fake_answer(question, chunks, prompt_builder, model):
        generated_with.append(model)
        return {"answer": "respuesta", "sources": []}

    def fake_judge(question, context, answer, model):
        return {"relevance": "RELEVANT", "grounded": True, "explanation": "ok"}

    monkeypatch.setattr(evaluate_llm, "answer_question", fake_answer)
    monkeypatch.setattr(evaluate_llm, "judge_answer", fake_judge)
    monkeypatch.setattr(evaluate_llm, "throttle", lambda: None)

    evaluate_llm.evaluate_prompt(
        "prompt_a",
        ["¿Qué es la tristeza bovina?"],
        sample_chunks,
        results,
        tmp_path / "results.json",
        generation_model="openai/gpt-oss-120b",
        judge_model="openai/gpt-oss-120b",
    )

    assert generated_with == ["openai/gpt-oss-120b"]
    new_run = get_run(results, "prompt_a", "openai/gpt-oss-120b", "openai/gpt-oss-120b")
    assert new_run["judgments"]["¿Qué es la tristeza bovina?"]["relevance"] == "RELEVANT"
    assert (tmp_path / "results.json").exists()


def test_summarize_reports_rates_with_intervals():
    judgments = {
        "q1": {"relevance": "RELEVANT", "grounded": True},
        "q2": {"relevance": "PARTLY_RELEVANT", "grounded": True},
    }
    summary = summarize(judgments, ["q1", "q2", "q3"])
    assert summary["evaluated"] == 2
    assert summary["relevant_pct"] == 0.5
    assert summary["grounded_pct"] == 1.0
    assert summary["relevant_ci"][0] < 0.5 < summary["relevant_ci"][1]
