# Engineering Improvements (v0.2.0)

This document maps each change made after the project review to the
software-engineering principle that motivated it. It doubles as a guide for
presenting the project.

## 1. Information hiding (Parnas, 1972)

> Isolate each design decision that is likely to change inside one module, so
> a change does not propagate through the system.

| Decision likely to change | Before | After |
|---|---|---|
| Which LLM model to use | Hardcoded in `groq_client.py` **and** `judge.py` | Read only in `src/config.py`, from `.env` |
| How to call the LLM provider | Each caller built its own Groq request | Only `src/generation/groq_client.complete()` knows the Groq SDK |
| Paths, `top_k` | Constants repeated in `main.py` and 3 scripts | `src/config.py` |

When Groq retired `llama-3.3-70b-versatile`, the old design needed code edits
in two files. Now it is a one-line `.env` change.

## 2. Low coupling

| Change | Coupling reduced |
|---|---|
| `rag.py` and `judge.py` call `complete()` instead of the Groq SDK | Generation logic no longer depends on a third-party SDK's types and errors |
| Groq exceptions are translated into `ModelUnavailableError` / `LLMServiceUnavailableError` | `main.py` and the evaluation never import `groq` |
| The chat UI talks to the API only over HTTP (`API_URL`) | UI and back end can change, scale or be deployed independently (message coupling, the lowest level) |
| The dashboard reuses `db.get_connection()` | One connection policy instead of two copies |

## 3. High cohesion / Single Responsibility (SRP)

| Module | Its one reason to change |
|---|---|
| `src/config.py` | How settings are provided |
| `src/jsonl.py` | The JSON Lines format (it was duplicated in 4 files) |
| `src/ingestion/pdf_loader.py` | How PDFs are read (file I/O helpers moved out) |
| `src/evaluation/stats.py` | How uncertainty is quantified |
| `src/evaluation/retrieval.py` | Retrieval metrics (the evaluation script had a second copy of this logic) |
| `src/ui/chat_app.py` | Presentation to the producer |

The evaluation scripts moved from `src/tests/` (they were not tests) to
`src/evaluation/`, next to the metrics they use; real tests live in `tests/`.

## 4. Open/Closed and Dependency Inversion (SOLID)

* **Open/Closed**: `answer_question(..., prompt_builder=...)` and
  `model=...` let new prompt variants and models be evaluated without
  modifying the generation function.
* **Dependency inversion**: high-level modules (API, RAG, evaluation) depend on
  a small abstraction (`complete()` returning a `Completion`), not on the Groq
  client. Tests replace it with a fake.

## 5. Verification and testing

* **54 automated tests** with `pytest` (plus one intentional `xfail`), offline and free.
* **Regression tests** for both reviewer findings: a decommissioned model must
  give a 503 with instructions, and a model change must not replay cached
  judgments.
* **A test found a real bug**: chunk overlaps can start mid-word. It is
  documented as `xfail` until it can be fixed together with re-ingestion.
* **Validation of the evaluation itself**: the ground truth grew from 10 to 42
  questions across all documents, and results now carry 95% confidence
  intervals, so "100%" on 10 questions is no longer mistaken for certainty.

## 6. Usability: Nielsen's 10 heuristics in the chat UI

| # | Heuristic | Where |
|---|---|---|
| H1 | Visibility of system status | Spinner while answering; "service online" badge with the model in use; toast confirming feedback |
| H2 | Match with the real world | Spanish, producer vocabulary; citations show report titles and pages instead of file names |
| H3 | User control and freedom | "Nueva conversación" button |
| H4 | Consistency and standards | Same green/yellow theme as the monitoring dashboard; standard chat layout |
| H5 | Error prevention | Input disabled when the API is down; 500-character limit; example questions |
| H6 | Recognition rather than recall | Clickable example questions; list of available documents |
| H7 | Flexibility and efficiency | Free text for experts, one-click examples for newcomers |
| H8 | Aesthetic and minimalist design | Only the conversation on screen; sources folded in an expander; developer menu hidden |
| H9 | Help users recover from errors | Plain-language messages for timeouts, connection errors and model outages |
| H10 | Help and documentation | "¿Cómo funciona?" panel and the veterinary disclaimer |

## 7. Reproducibility

* `.env.example` documents every variable; the README has a step-by-step guide
  for future model deprecations.
* `/health` exposes the configured model; every conversation records the model
  that answered it, and the dashboard charts it.
* Docker Compose now also starts the chat UI (`:8502`).
