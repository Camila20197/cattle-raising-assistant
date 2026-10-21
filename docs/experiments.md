# Experiment Log and Theoretical Notes

This document records the experiments performed while building the Cattle Health Assistant. It will be updated whenever the retrieval pipeline changes.

## Project Goal

The application is a Retrieval-Augmented Generation (RAG) assistant for questions about cattle health and management. It retrieves evidence from the supplied INTA PDF documents before a language model generates an answer.

The intended flow is:

```text
User question -> retrieve relevant chunks -> build an evidence-based prompt -> LLM answer with sources
```

The assistant is informational and does not replace professional veterinary advice.

## Theory Overview

### RAG

A language model does not automatically know the content of the local INTA PDFs. RAG supplies relevant source text at question time. This helps ground the answer in the project corpus and makes it possible to cite the original document and page.

### Chunking

PDF pages are usually too long to retrieve as single units. Chunking divides a page into smaller overlapping pieces. The overlap preserves context when an idea crosses a chunk boundary. Each chunk keeps its document name and page number as metadata.

### Keyword Retrieval (BM25)

Keyword retrieval ranks a chunk by how well its words match the query. An early version of this project scored chunks by simple term-frequency counting (how many times each query word appeared), which was later found to have a significant flaw — see Experiment 1. The corrected implementation uses **BM25**, which additionally weighs each word by how rare it is across the whole corpus (inverse document frequency) and saturates the contribution of a word that repeats many times within one chunk. Stop words such as `la`, `de`, and `para` are removed before scoring in both versions.

### Vector Retrieval

An embedding model converts text into a numeric vector that represents semantic meaning. A question and each document chunk are embedded in the same vector space. Cosine similarity ranks chunks whose vectors point in a similar direction to the question vector.

For this project, Gemini Embedding 2 creates 768-dimensional vectors. Document and question embeddings use different task-oriented prefixes, following the Gemini retrieval guidance.

### Retrieval Metrics

* **Hit Rate@k** is the proportion of evaluation questions for which the expected chunk is found in the top `k` results.
* **Mean Reciprocal Rank (MRR)@k** also measures position. A correct result at rank 1 contributes `1`, at rank 2 contributes `1/2`, and a missing result contributes `0`.

Higher values are better. MRR is useful when two methods find the same number of correct chunks but rank them differently.

## Dataset and Ingestion

| Item | Result |
| --- | ---: |
| Source PDF documents | 4 |
| Total extracted pages | 64 |
| Empty pages skipped | 2 |
| Text pages available for retrieval | 62 |

The empty pages were the first and last pages of the tick-identification guide. They contained no extractable text, so they were excluded from the text retrieval corpus.

Page records are stored in `data/processed/pages.jsonl`. Each record contains:

```json
{"document": "source.pdf", "page": 1, "text": "..."}
```

### Ingestion automation

The pipeline that produces `pages.jsonl` -> `chunks.jsonl` -> `indexed_chunks.jsonl` was originally run by hand, cell by cell, in a notebook. It was consolidated into a single non-interactive script, `src/ingestion/run_ingestion.py`, and automated with **Kestra**: a flow (`cattle_health_ingestion`) runs this script inside the project's own Docker image, sharing a Docker volume with the API service so both read and write the same processed files. The flow can be triggered manually from the Kestra UI or on a weekly schedule.

A full automated run reproduced the same dataset in ~76 seconds: 62 pages extracted, 192 chunks created, 192 chunks embedded — confirming the pipeline is repeatable without manual intervention.

## Chunking Experiment

| Parameter | Value |
| --- | ---: |
| Chunk size | 1,000 characters |
| Overlap | 200 characters |
| Chunks created | 192 |
| Largest chunk observed | 999 characters |

Chunks are stored in `data/processed/chunks.jsonl` and keep `id`, `document`, `page`, `chunk_index`, and `text`.

## Retrieval Evaluation Set

The manually verified ground-truth set is stored in `data/evaluation/retrieval_ground_truth.json`.

| Item | Value |
| --- | ---: |
| Questions | 10 |
| Expected evidence | Chunk IDs manually checked against the INTA text |
| Evaluation cutoff | Top 5 results (`@5`) |

The set covers ivermectin and bovine alphaherpesvirus, drought impacts, and ticks.

## Experiment 1: Keyword Retrieval — bug found, then fixed with BM25

**Initial method:** raw term-frequency counting — a chunk's score was the sum of how many times each query word appeared in it, with no weighting.

**Bug found:** this scored generic, corpus-wide terms (e.g. "bovinos", present in nearly every document) with the same weight as rare, distinctive terms (e.g. "sequía"). A manual test with the query *"¿Qué efectos tuvo la sequía sobre la ganancia de peso de los bovinos?"* returned a tick-identification document ranked above the drought-impact document that actually answered the question — the clearest sign the scoring was rewarding the wrong signal.

**Fix:** replaced term-frequency counting with **BM25** (`rank_bm25` library), which weighs each term by its rarity across the corpus (IDF) and saturates repeated-term contributions.

**Result (BM25, final):**

| Metric | Value |
| --- | ---: |
| Hit Rate@5 | 100.00% |
| MRR@5 | 81.70% |

After the fix, keyword search found the expected chunk for every question, almost always at rank 1.

## Experiment 2: Gemini Vector Retrieval

**Method:** Gemini Embedding 2, 768 dimensions, cosine similarity, in-memory index.

**Indexing details:**

* 192 chunks were embedded and saved in `data/processed/indexed_chunks.jsonl`.
* The free-tier request limit required embedding in batches of 49 chunks with a pause before starting a new quota window.
* Vectors are normalized before similarity scoring.

**Result:**

| Metric | Value |
| --- | ---: |
| Hit Rate@5 | 90.00% |
| MRR@5 | 58.70% |

The vector approach found the expected chunk for 9 of 10 questions, but ranked correct chunks lower on average than BM25.

## Experiment 3: Hybrid Retrieval

**Method:** keyword (BM25) and vector retrieval combined with Reciprocal Rank Fusion (RRF).

RRF combines the positions of results rather than their raw scores. This is important because BM25 scores and cosine-similarity scores have different scales and should not be added directly.

**Result:**

| Metric | Value |
| --- | ---: |
| Hit Rate@5 | 100.00% |
| MRR@5 | 77.00% |

Hybrid found the expected chunk for every question, but ranked slightly worse on average than BM25 alone. RRF weighs the keyword and vector rankings equally when fusing them; since vector search places the correct chunk lower on average, the fused ranking gets pulled down relative to keyword search by itself. No manual re-weighting of the fusion was attempted, since the goal of this stage was to measure, not to optimize further.

## Current Conclusion — Retrieval

No retrieval method should be selected only because it is more sophisticated. On the final evaluation set, **BM25 keyword search has the best MRR** and is the method used in production, despite hybrid search also achieving a perfect Hit Rate:

| Method | Hit Rate@5 | MRR@5 |
| --- | ---: | ---: |
| Keyword (BM25) | 100.00% | **81.70%** |
| Vector (Gemini embeddings) | 90.00% | 58.70% |
| Hybrid (RRF) | 100.00% | 77.00% |

This is a domain-specific finding: the ground-truth questions use precise technical vocabulary (species names, disease terms) that tends to match the source text literally, favoring lexical search over semantic search in this corpus. Hybrid search was evaluated and satisfies the project's hybrid-search best-practice criterion, but it was not the method selected for production — the decision was based on the measured metric, not on which method is more elaborate.

## Experiment 4: Grounded Answer Generation

**Method:** the top chunks from BM25 retrieval are formatted into a prompt and sent to an LLM. The prompt requires Spanish output, evidence-based factual claims, document-page citations, and a veterinary safety notice.

**Prompt safeguards:**

* The model must use only the retrieved context.
* The model must state when the context is insufficient.
* The model must not diagnose animals or give personalized treatment or dosage instructions.
* Each factual claim must include a document-page citation, in the exact format `[document_name.pdf: page_number]`.

**Generation provider:** initially Gemini 3.5 Flash; later switched to **Groq (`llama-3.3-70b-versatile`)** for both generation and judging, to work around the Gemini free tier's low daily request quota (20/day on this project) encountered during evaluation. Gemini was kept for embeddings only, since Groq does not offer an embeddings endpoint. `temperature=0.1` for near-deterministic, reproducible answers; `max_output_tokens=2048`.

**Truncation bug found and fixed:** a manual test through the FastAPI interface returned an answer cut off mid-sentence. The cause was `max_output_tokens` set too low relative to the model's internal reasoning budget. The fix was twofold: raising the token budget, and explicitly checking the response's `finish_reason` so a truncated response is reported as such instead of silently returned as if complete.

This verifies that the RAG flow can generate a cited, safety-compliant answer end to end, including through the containerized API.

## Experiment 5: LLM Evaluation — Prompt A vs Prompt B

**Method:** LLM-as-a-judge. For each of the 10 ground-truth questions, both prompt variants generate an answer (using BM25-retrieved context), and a judge model (Groq, `llama-3.3-70b-versatile`, `temperature=0`) scores each answer on two axes:

* **Relevance**: `RELEVANT` / `PARTLY_RELEVANT` / `NON_RELEVANT` — does the answer address the question?
* **Groundedness**: `true`/`false` — is every factual claim backed by the retrieved context, with nothing invented?

**Prompts compared:**

* **Prompt A (detailed):** explicit numbered rules, with a worked citation example.
* **Prompt B (concise):** the same constraints (Spanish, grounding, safety, citation format) written as a single short paragraph, without the worked example.

**Result:**

| Prompt | Relevant | Grounded |
| --- | ---: | ---: |
| A (detailed) | 90.00% | 100.00% |
| B (concise) | 90.00% | 100.00% |

The two prompts tied exactly. **Prompt A was kept for production**: with equal measured quality on this sample, the worked citation example is expected to generalize better to questions outside the 10-question set, where citation formatting is more likely to drift without an anchoring example. One question was not answered relevantly by either prompt — recorded here as a known limitation rather than a case worth over-fitting the prompt to, given the small sample size.

**Known limitation of this evaluation:** the judge scores relevance and groundedness, but not answer *completeness* — the truncation bug in Experiment 4 was caught by manual testing, not by this automated evaluation, since a truncated-but-otherwise-grounded partial answer would not necessarily be flagged by the current judge prompt.

## Experiment 6: Post-review hardening (v0.2.0)

Changes made after the project review, each tied to a concrete finding:

**1. Model deprecation broke reproducibility.** Groq retired `llama-3.3-70b-versatile` on 2026-08-16 ([Groq deprecations](https://console.groq.com/docs/deprecations)), so `POST /ask` returned an opaque 500. Fixes:

* The model name moved out of the code into configuration (`GROQ_GENERATION_MODEL`, `GROQ_JUDGE_MODEL`, read by `src/config.py`). The default is now `openai/gpt-oss-120b`, Groq's recommended replacement.
* All Groq calls go through one adapter (`src/generation/groq_client.complete`). It translates a retired/unknown model into `ModelUnavailableError`, which the API returns as **503 with instructions** (which variable to change, where to check deprecations) instead of a 500.
* `/health` and every `/ask` response report the model in use, and the model is logged in the `conversations` table and shown on the monitoring dashboard.
* `GENERATION_MAX_TOKENS` defaults to 4096 because gpt-oss is a reasoning model and spends part of the budget reasoning (same truncation risk as Experiment 4).

**2. The LLM-evaluation cache replayed stale judgments.** Results were keyed only by question text, so after swapping the model the script silently reused judgments produced by the retired model. Results are now keyed by `prompt | generation model | judge model` (results file format v2). The old judgments were migrated under `llama-3.3-70b-versatile`, kept as history, and are never reused for another model. A regression test (`tests/test_llm_eval_cache.py`) covers this exact scenario.

**3. The evaluation set was too small for statistical confidence.** The ground truth grew from 10 to **42 questions**, now covering all four documents (the alfaherpesvirus-in-ticks paper had no questions before, and the drought report had only one). The 32 new questions are marked `"authoring": "ai_assisted"`: they were drafted with AI assistance and automatically checked so that every expected chunk contains the key fact of the answer. Every metric is now reported with a 95% confidence interval (Wilson for proportions, bootstrap for MRR).

BM25 on the expanded set (`uv run python -m src.evaluation.evaluate_retrieval --methods keyword`):

| Ground truth | Hit Rate@5 | 95% CI | MRR | 95% CI |
| --- | ---: | --- | ---: | --- |
| Original 10 questions | 100% | [72%, 100%] | 0.817 | — |
| Expanded 42 questions | 90.5% | [78%, 96%] | 0.724 | [0.61, 0.83] |

The larger set gives a lower but more trustworthy estimate: the original 100% was compatible with a true hit rate as low as 72%. The four misses (e.g. *"¿Qué vacuna comercial se usó como comparación en el ensayo de ivermectina?"*) point at real weaknesses (questions whose key term is a brand name or number that BM25 does not weight well). Vector and hybrid search, and the LLM evaluation with the new model, need to be re-run with API keys to complete the comparison on 42 questions.

**4. Logs and evaluation output were in Spanish.** All logs, script output and evaluation reports are now in English (via the `logging` module). The assistant's answers and the chat UI stay in Spanish on purpose, because the end users are Argentine producers.

**5. A test found a chunking bug.** The new unit tests showed that the overlap between consecutive chunks can start in the middle of a word (`"tres"` → `"res"`); about 80% of non-first chunks start this way. The fix changes chunk boundaries and therefore the chunk IDs used by the ground truth, so it is tracked as an `xfail` test and in the roadmap, to be shipped together with a re-ingestion and a regenerated ground truth.

## Interface and Deployment

* **Interface:** a FastAPI application (`main.py`) exposes `POST /ask` (question in, grounded answer + sources out) and `GET /health` (a liveness check that makes no external API calls). The root path redirects to the auto-generated `/docs` Swagger UI.
* **Chat UI:** a Streamlit chat (`src/ui/chat_app.py`, port 8502) consumes the API over HTTP, designed with Nielsen's usability heuristics (see `docs/engineering-improvements.md`).
* **Containerization:** the API, Postgres, the dashboard, the chat UI and Kestra run via `docker-compose.yml`. The API image is built from a `Dockerfile` using `uv sync --frozen` for reproducible dependency installation. A shared named Docker volume (`app_data`) lets the Kestra-run ingestion flow and the API read and write the same processed data files.
* **Reproducibility:** both API keys (`GEMINI_API_KEY` for embeddings, `GROQ_API_KEY` for generation/judging) are supplied via a local `.env` file, never baked into the Docker image.

## References

* [Gemini Embeddings documentation](https://ai.google.dev/gemini-api/docs/embeddings)
* [Groq API Reference](https://console.groq.com/docs/api-reference)
* [rank_bm25 (PyPI)](https://pypi.org/project/rank-bm25/)
* [Kestra Docker Task Runner documentation](https://kestra.io/docs/task-runners/types/docker-task-runner)
* [LLM Zoomcamp project evaluation criteria](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/project.md#evaluation-criteria)
