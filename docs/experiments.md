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

### Keyword Retrieval

Keyword retrieval ranks a chunk by the number of meaningful query words that occur in it. It is fast and transparent, but it cannot recognize synonyms or semantic similarity. Stop words such as `la`, `de`, and `para` are removed before scoring.

### Vector Retrieval

An embedding model converts text into a numeric vector that represents semantic meaning. A question and each document chunk are embedded in the same vector space. Cosine similarity ranks chunks whose vectors point in a similar direction to the question vector.

For this project, Gemini Embedding 2 creates 768-dimensional vectors. Document and question embeddings use different task-oriented prefixes, following the Gemini retrieval guidance.

### Retrieval Metrics

* **Hit Rate@k** is the proportion of evaluation questions for which at least one expected chunk is found in the top `k` results.
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

The first version contained five questions. It was expanded to ten questions covering ivermectin and bovine alphaherpesvirus, drought impacts, and ticks. It must be expanded further before the final project evaluation.

## Experiment 1: Keyword Retrieval

**Method:** token matching with Spanish stop-word removal.

**Result:**

| Metric | Value |
| --- | ---: |
| Hit Rate@5 | 80.00% |
| MRR@5 | 66.67% |

Four of the five expected chunks were found in the top five results. The results showed that removing stop words improves ranking by preventing generic words from dominating the score.

## Experiment 2: Gemini Vector Retrieval

**Method:** Gemini Embedding 2, 768 dimensions, cosine similarity, in-memory index.

**Indexing details:**

* 192 chunks were embedded and saved in `data/processed/indexed_chunks.jsonl`.
* The free-tier request limit required embedding in batches of 49 chunks with a pause before starting a new quota window.
* Vectors are normalized before similarity scoring.

**Result:**

| Metric | Value |
| --- | ---: |
| Hit Rate@5 | 80.00% |
| MRR@5 | 54.00% |

The vector approach found evidence for the same four questions, but placed correct chunks lower in the ranking. Therefore, the keyword baseline is currently the better individual retrieval method according to MRR.

## Experiment 3: Hybrid Retrieval

**Method:** keyword retrieval and vector retrieval combined with Reciprocal Rank Fusion (RRF).

RRF combines the positions of results rather than their raw scores. This is important because keyword-match counts and cosine-similarity scores have different scales and should not be added directly.

**Result:**

| Metric | Value |
| --- | ---: |
| Hit Rate@5 | 100.00% |
| MRR@5 | 80.00% |

All five expected chunks were found in the top five results. The correct chunk ranks were 1, 1, 2, 1, and 2.

## Experiment 4: Hybrid Retrieval with Expanded Ground Truth

**Method:** the same hybrid retrieval configuration evaluated with the expanded ten-question ground-truth set.

**Result:**

| Metric | Value |
| --- | ---: |
| Hit Rate@5 | 100.00% |
| MRR@5 | 90.00% |

The correct chunks ranked 1, 1, 2, 1, 2, 1, 1, 1, 1, and 1. This confirms that the hybrid method remains effective beyond the original five questions.

## Current Conclusion

No retrieval method should be selected only because it is more sophisticated. On the current evaluation set, hybrid retrieval has the best result and will be the selected method for the RAG application.

| Method | Hit Rate@5 | MRR@5 |
| --- | ---: | ---: |
| Keyword | 80.00% | 66.67% |
| Vector | 80.00% | 54.00% |
| Hybrid, initial 5-question set | **100.00%** | 80.00% |
| Hybrid, expanded 10-question set | **100.00%** | **90.00%** |

The next improvement is to expand the manually verified evaluation set further and then begin evaluating answer generation.

## Experiment 5: Grounded Answer Generation

**Method:** the top three chunks from hybrid retrieval are formatted into a prompt and sent to Gemini 3.5 Flash. The prompt requires Spanish output, evidence-based factual claims, document-page citations, and a veterinary safety notice.

**Prompt safeguards:**

* The model must use only the retrieved context.
* The model must state when the context is insufficient.
* The model must not diagnose animals or give personalized treatment or dosage instructions.
* Each factual claim must include a document-page citation.

**Configuration:** `max_output_tokens=1000` and `thinking_level="low"`.

The first test used the default thinking level with 500 output tokens and returned a truncated answer. Increasing the output limit and selecting low thinking produced a complete answer to the question about conventional BoAHV-1 control measures. The answer correctly cited vaccination, hygiene measures, and quarantine for new animals from page 2 of the source document.

This verifies that the RAG flow can generate a cited answer. It is not yet an LLM evaluation: the next step is to create a question-and-answer evaluation set and compare answer quality across prompt or model configurations.

## References

* [Gemini Embeddings documentation](https://ai.google.dev/gemini-api/docs/embeddings)
* [LLM Zoomcamp project evaluation criteria](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/project.md#evaluation-criteria)
