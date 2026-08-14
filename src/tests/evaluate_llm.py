"""Compare two generation prompts (A: detailed, B: concise) using an LLM judge.

For a sample of questions from the ground truth, retrieves context with BM25
keyword search (the best-performing retrieval method per
evaluate_retrieval.py) and generates an answer with both prompt variants. An
LLM judge then scores each answer for relevance and groundedness, and the
script reports the percentage of RELEVANT and grounded answers per prompt.

Resilience:
- Each per-question step is retried with backoff on transient server errors
  (5xx), and paced with a proactive rate limiter to respect the per-minute
  quota.
- Results are written to disk after EVERY question (not just at the end),
  so a crash never discards already-computed (already-paid-for) judgments.
- The script is resumable: on start, it loads whatever is already in
  RESULTS_PATH and skips any (prompt, question) pair already judged. If you
  hit the free-tier DAILY quota (a hard stop, not something retries can
  fix), just wait for the daily reset and re-run the same command — it will
  pick up exactly where it left off, at zero extra cost.

NOTE: adjust the two imports below to match where you placed rag.py and
judge.py in your project (this assumes src/generation/).
"""

import json
import time
from collections import deque
from pathlib import Path
from typing import Callable

from groq import APIConnectionError, InternalServerError, RateLimitError

from src.generation.rag import answer_question, build_prompt, build_prompt_concise, format_context
from src.generation.judge import judge_answer
from src.retrieval.keyword_search import search_chunks

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
GROUND_TRUTH_PATH = Path("data/evaluation/retrieval_ground_truth.json")
RESULTS_PATH = Path("data/evaluation/llm_eval_results.json")
TOP_K = 5
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 15

# Groq's free-tier daily cap is far higher than Gemini's -- the full 10
# questions fit comfortably. Kept as a named constant so it's still one
# place to change if you tune it later.
EVAL_SAMPLE_SIZE = 10

# Free-tier RPM limit for gemini-3.5-flash generateContent (separate from
# the daily cap above -- both apply, and both matter).
RATE_LIMIT_CALLS_PER_MINUTE = 5
RATE_LIMIT_WINDOW_SECONDS = 65  # 60s window + a safety margin

_recent_call_times: deque[float] = deque()


def throttle() -> None:
    """Block until another generateContent call is safe under the RPM limit."""
    now = time.monotonic()
    while _recent_call_times and now - _recent_call_times[0] > RATE_LIMIT_WINDOW_SECONDS:
        _recent_call_times.popleft()

    if len(_recent_call_times) >= RATE_LIMIT_CALLS_PER_MINUTE:
        wait_time = RATE_LIMIT_WINDOW_SECONDS - (now - _recent_call_times[0])
        if wait_time > 0:
            print(f"    Esperando {wait_time:.0f}s para no superar "
                  f"{RATE_LIMIT_CALLS_PER_MINUTE} req/min...")
            time.sleep(wait_time)

    _recent_call_times.append(time.monotonic())


def call_with_retry(func: Callable, *args, **kwargs):
    for attempt in range(1, MAX_RETRIES + 1):
        throttle()
        try:
            return func(*args, **kwargs)
        except RateLimitError as error:
            if attempt == MAX_RETRIES:
                raise
            print(f"    Cuota alcanzada, esperando {RATE_LIMIT_WINDOW_SECONDS}s "
                  f"(intento {attempt}/{MAX_RETRIES})...")
            time.sleep(RATE_LIMIT_WINDOW_SECONDS)
        except (InternalServerError, APIConnectionError) as error:
            if attempt == MAX_RETRIES:
                raise
            print(f"    Error transitorio ({error}), reintentando en {RETRY_DELAY_SECONDS}s "
                  f"(intento {attempt}/{MAX_RETRIES})...")
            time.sleep(RETRY_DELAY_SECONDS)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def load_questions(path: Path, sample_size: int) -> list[str]:
    with path.open("r", encoding="utf-8") as file:
        ground_truth = json.load(file)
    return [item["question"] for item in ground_truth[:sample_size]]


def load_results() -> dict[str, dict[str, dict]]:
    """Load previously computed judgments, if any, so the run can resume."""
    if not RESULTS_PATH.exists():
        return {"prompt_a": {}, "prompt_b": {}}
    with RESULTS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_results(results: dict[str, dict[str, dict]]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)


def evaluate_prompt(
    prompt_key: str,
    prompt_builder: Callable,
    questions: list[str],
    chunks: list[dict],
    results: dict[str, dict[str, dict]],
) -> None:
    """Judge every question for one prompt variant, skipping already-cached ones.

    Mutates ``results`` in place and saves to disk after each new judgment,
    so progress always survives a crash or a daily-quota cutoff.
    """
    already_done = results[prompt_key]

    for position, question in enumerate(questions, start=1):
        if question in already_done:
            print(f"  [{position}/{len(questions)}] ya calculada, se salta")
            continue

        retrieved = search_chunks(question, chunks, top_k=TOP_K)
        result = call_with_retry(
            answer_question, question, retrieved, prompt_builder=prompt_builder
        )
        context = format_context(retrieved)
        judgment = call_with_retry(judge_answer, question, context, result["answer"])

        already_done[question] = judgment
        save_results(results)
        print(f"  [{position}/{len(questions)}] ok (guardado)")


def summarize(results: dict[str, dict]) -> dict[str, float | int]:
    values = list(results.values())
    total = len(values)
    if total == 0:
        return {"total_evaluadas": 0, "relevant_pct": 0.0, "grounded_pct": 0.0}

    relevant = sum(1 for judgment in values if judgment["relevance"] == "RELEVANT")
    grounded = sum(1 for judgment in values if judgment["grounded"])

    return {
        "total_evaluadas": total,
        "relevant_pct": relevant / total,
        "grounded_pct": grounded / total,
    }


def print_report(summary_a: dict[str, float], summary_b: dict[str, float]) -> None:
    print(f"\n{'Prompt':<16}{'Evaluadas':<12}{'Relevant':<12}{'Grounded':<12}")
    print("-" * 52)
    print(f"{'A (detallado)':<16}{summary_a['total_evaluadas']:<12}"
          f"{summary_a['relevant_pct']:<12.1%}{summary_a['grounded_pct']:<12.1%}")
    print(f"{'B (conciso)':<16}{summary_b['total_evaluadas']:<12}"
          f"{summary_b['relevant_pct']:<12.1%}{summary_b['grounded_pct']:<12.1%}")

    if summary_a["total_evaluadas"] and summary_b["total_evaluadas"]:
        best = "A (detallado)" if summary_a["grounded_pct"] >= summary_b["grounded_pct"] else "B (conciso)"
        print(f"\nMejor prompt según groundedness: {best}")
    else:
        print("\n(Todavía faltan preguntas por evaluar en algún prompt -- volvé a correr el script.)")


def main():
    chunks = load_jsonl(CHUNKS_PATH)
    questions = load_questions(GROUND_TRUTH_PATH, EVAL_SAMPLE_SIZE)
    results = load_results()

    try:
        print(f"Evaluando Prompt A ({len(questions)} preguntas)...")
        evaluate_prompt("prompt_a", build_prompt, questions, chunks, results)

        print(f"\nEvaluando Prompt B ({len(questions)} preguntas)...")
        evaluate_prompt("prompt_b", build_prompt_concise, questions, chunks, results)
    except RateLimitError:
        print("\nSe alcanzó un límite de cuota que no se resolvió tras los reintentos. Lo ya "
              f"calculado quedó guardado en {RESULTS_PATH}. Esperá un momento y volvé a correr "
              "este mismo comando -- va a continuar desde donde quedó.")
        return

    print_report(summarize(results["prompt_a"]), summarize(results["prompt_b"]))


if __name__ == "__main__":
    main()