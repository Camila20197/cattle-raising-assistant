"""Compare prompt A vs prompt B with an LLM-as-a-judge.

For each ground-truth question and each prompt variant, the assistant
generates an answer (BM25 context) and a judge model scores it for
relevance and groundedness. Results are reported with 95% Wilson intervals.

The run is resumable: every judgment is saved to disk as soon as it is
produced. The cache key includes the prompt variant AND both model names,
so changing GROQ_GENERATION_MODEL or GROQ_JUDGE_MODEL starts a fresh run
instead of silently replaying judgments produced by another (possibly
retired) model.

Run with:
    uv run python -m src.evaluation.evaluate_llm
    uv run python -m src.evaluation.evaluate_llm --sample-size 10
"""

import argparse
import json
import logging
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

from src.config import get_settings
from src.evaluation.stats import format_interval, wilson_interval
from src.generation.groq_client import LLMServiceUnavailableError
from src.generation.judge import judge_answer
from src.generation.rag import answer_question, build_prompt, build_prompt_concise, format_context
from src.jsonl import read_jsonl
from src.retrieval.keyword_search import build_bm25_index, search_chunks

logger = logging.getLogger(__name__)

RESULTS_FORMAT_VERSION = 2
# Model used for every judgment stored before results were keyed by model.
LEGACY_MODEL = "llama-3.3-70b-versatile"

PROMPTS: dict[str, tuple[str, Callable]] = {
    "prompt_a": ("A (detailed)", build_prompt),
    "prompt_b": ("B (concise)", build_prompt_concise),
}

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 65
RATE_LIMIT_CALLS_PER_MINUTE = 5
RATE_LIMIT_WINDOW_SECONDS = 65

_recent_call_times: deque[float] = deque()


# --------------------------------------------------------------------------
# Result cache
# --------------------------------------------------------------------------

def cache_key(prompt_key: str, generation_model: str, judge_model: str) -> str:
    """Identify one evaluation run: same prompt + same models = same results."""
    return f"{prompt_key}|gen={generation_model}|judge={judge_model}"


def empty_results() -> dict:
    return {"format_version": RESULTS_FORMAT_VERSION, "runs": {}}


def migrate_legacy_results(legacy: dict) -> dict:
    """Convert the old ``{"prompt_a": {...}, "prompt_b": {...}}`` file.

    Old judgments are kept under the model that actually produced them, so
    they remain available as history but are never reused for a new model.
    """
    results = empty_results()
    for prompt_key, judgments in legacy.items():
        if prompt_key in PROMPTS and isinstance(judgments, dict):
            results["runs"][cache_key(prompt_key, LEGACY_MODEL, LEGACY_MODEL)] = {
                "prompt": prompt_key,
                "generation_model": LEGACY_MODEL,
                "judge_model": LEGACY_MODEL,
                "judgments": judgments,
            }
    return results


def load_results(path: Path) -> dict:
    if not path.exists():
        return empty_results()
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if data.get("format_version") == RESULTS_FORMAT_VERSION:
        return data
    logger.info("Migrating legacy results file %s to format v%s", path, RESULTS_FORMAT_VERSION)
    return migrate_legacy_results(data)


def save_results(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)


def get_run(results: dict, prompt_key: str, generation_model: str, judge_model: str) -> dict:
    """Return (creating it if needed) the run for this prompt + model pair."""
    key = cache_key(prompt_key, generation_model, judge_model)
    return results["runs"].setdefault(
        key,
        {
            "prompt": prompt_key,
            "generation_model": generation_model,
            "judge_model": judge_model,
            "judgments": {},
        },
    )


# --------------------------------------------------------------------------
# Rate limiting and retries
# --------------------------------------------------------------------------

def throttle() -> None:
    """Block until another API call is safe under the requests-per-minute limit."""
    now = time.monotonic()
    while _recent_call_times and now - _recent_call_times[0] > RATE_LIMIT_WINDOW_SECONDS:
        _recent_call_times.popleft()

    if len(_recent_call_times) >= RATE_LIMIT_CALLS_PER_MINUTE:
        wait_time = RATE_LIMIT_WINDOW_SECONDS - (now - _recent_call_times[0])
        if wait_time > 0:
            logger.info(
                "Waiting %.0fs to stay under %s requests/minute...",
                wait_time,
                RATE_LIMIT_CALLS_PER_MINUTE,
            )
            time.sleep(wait_time)

    _recent_call_times.append(time.monotonic())


def call_with_retry(func: Callable, *args, **kwargs):
    """Retry temporary provider errors; let configuration errors fail fast."""
    for attempt in range(1, MAX_RETRIES + 1):
        throttle()
        try:
            return func(*args, **kwargs)
        except LLMServiceUnavailableError as error:
            if attempt == MAX_RETRIES:
                raise
            logger.warning(
                "%s -- retrying in %ss (attempt %s/%s)",
                error,
                RETRY_DELAY_SECONDS,
                attempt,
                MAX_RETRIES,
            )
            time.sleep(RETRY_DELAY_SECONDS)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def evaluate_prompt(
    prompt_key: str,
    questions: list[str],
    chunks: list[dict],
    results: dict,
    results_path: Path,
    generation_model: str,
    judge_model: str,
) -> None:
    """Judge every question for one prompt variant, skipping cached ones.

    Saves to disk after each new judgment, so progress survives a crash or
    a daily-quota cutoff.
    """
    _, prompt_builder = PROMPTS[prompt_key]
    run = get_run(results, prompt_key, generation_model, judge_model)
    judgments = run["judgments"]
    bm25 = build_bm25_index(chunks)
    top_k = get_settings().top_k

    for position, question in enumerate(questions, start=1):
        if question in judgments:
            logger.info("[%s/%s] already judged, skipping", position, len(questions))
            continue

        retrieved = search_chunks(question, chunks, top_k=top_k, bm25=bm25)
        if not retrieved:
            judgments[question] = {
                "relevance": "NON_RELEVANT",
                "grounded": True,
                "explanation": "No chunk was retrieved, so no answer was generated.",
            }
        else:
            result = call_with_retry(
                answer_question,
                question,
                retrieved,
                prompt_builder=prompt_builder,
                model=generation_model,
            )
            judgments[question] = call_with_retry(
                judge_answer,
                question,
                format_context(retrieved),
                str(result["answer"]),
                model=judge_model,
            )

        save_results(results, results_path)
        logger.info("[%s/%s] judged and saved", position, len(questions))


def summarize(judgments: dict[str, dict], questions: list[str]) -> dict:
    """Relevance and groundedness rates, with 95% intervals, for ``questions``."""
    values = [judgments[question] for question in questions if question in judgments]
    total = len(values)
    if total == 0:
        return {"evaluated": 0}

    relevant = sum(1 for judgment in values if judgment["relevance"] == "RELEVANT")
    grounded = sum(1 for judgment in values if judgment["grounded"])
    return {
        "evaluated": total,
        "relevant_pct": relevant / total,
        "relevant_ci": wilson_interval(relevant, total),
        "grounded_pct": grounded / total,
        "grounded_ci": wilson_interval(grounded, total),
    }


def print_report(summaries: dict[str, dict], generation_model: str, judge_model: str) -> None:
    print(f"\nLLM evaluation — generation: {generation_model} | judge: {judge_model}")
    print(f"{'Prompt':<15}{'Evaluated':<11}{'Relevant':<10}{'95% CI':<13}{'Grounded':<10}{'95% CI':<13}")
    print("-" * 72)
    for prompt_key, summary in summaries.items():
        label = PROMPTS[prompt_key][0]
        if summary["evaluated"] == 0:
            print(f"{label:<15}{0:<11}-")
            continue
        print(
            f"{label:<15}{summary['evaluated']:<11}"
            f"{summary['relevant_pct']:<10.1%}{format_interval(summary['relevant_ci']):<13}"
            f"{summary['grounded_pct']:<10.1%}{format_interval(summary['grounded_ci']):<13}"
        )
    print(
        "\nNote: with a few dozen questions the 95% intervals are wide; treat "
        "differences inside overlapping intervals as a tie."
    )


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="LLM-as-a-judge evaluation of prompt A vs B")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Evaluate only the first N ground-truth questions (default: all).",
    )
    parser.add_argument("--generation-model", default=settings.generation_model)
    parser.add_argument("--judge-model", default=settings.judge_model)
    args = parser.parse_args()

    chunks = read_jsonl(settings.chunks_path)
    with settings.ground_truth_path.open("r", encoding="utf-8") as file:
        ground_truth = json.load(file)
    questions = [item["question"] for item in ground_truth[: args.sample_size]]
    results = load_results(settings.llm_results_path)

    try:
        for prompt_key, (label, _) in PROMPTS.items():
            logger.info("Evaluating prompt %s on %s questions...", label, len(questions))
            evaluate_prompt(
                prompt_key,
                questions,
                chunks,
                results,
                settings.llm_results_path,
                args.generation_model,
                args.judge_model,
            )
    except LLMServiceUnavailableError:
        logger.error(
            "Provider limit not resolved after retries. Progress is saved in %s; "
            "wait a moment and re-run the same command to resume.",
            settings.llm_results_path,
        )
        return

    summaries = {
        prompt_key: summarize(
            get_run(results, prompt_key, args.generation_model, args.judge_model)["judgments"],
            questions,
        )
        for prompt_key in PROMPTS
    }
    print_report(summaries, args.generation_model, args.judge_model)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
