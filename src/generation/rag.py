"""Prompt construction and answer generation for the RAG application.

The user-facing answers stay in Spanish on purpose: the target users are
Argentine cattle producers. Code, logs and evaluation output are in English.
"""

from typing import Callable

from src.config import get_settings
from src.generation.groq_client import complete

NO_ANSWER_MESSAGE = "No se pudo generar una respuesta para esta pregunta. Intentá reformularla."
TRUNCATED_MESSAGE = "La respuesta se cortó antes de terminar. Probá reformular la pregunta de forma más específica."


def format_context(chunks: list[dict[str, str | int | float]]) -> str:
    """Format retrieved chunks with source metadata for a RAG prompt."""
    sections = []

    for index, chunk in enumerate(chunks, start=1):
        sections.append(
            "\n".join(
                [
                    f"[Source {index}]",
                    f"Document: {chunk['document']}",
                    f"Page: {chunk['page']}",
                    f"Content: {chunk['text']}",
                ]
            )
        )

    return "\n\n".join(sections)


def build_prompt(question: str, chunks: list[dict[str, str | int | float]]) -> str:
    """Build a grounded prompt using a question and retrieved source chunks."""
    context = format_context(chunks)

    return f"""Answer the user's question using only the source context below.

Rules:
- Answer in Spanish.
- Do not invent facts that are not present in the context.
- If the context is insufficient, say that the available documents do not contain enough information.
- Do not diagnose animals or provide personalized treatment or dosage instructions.
- Encourage consultation with a qualified veterinarian when the question concerns animal health.
- Cite every factual claim with its source, using this exact format: [document_name.pdf: page_number].
  Example: "Ivermectin reduced viral shedding [Maidana_ivermectina.pdf: 3]."

Source context:
{context}

User question: {question}
"""


def build_prompt_concise(question: str, chunks: list[dict[str, str | int | float]]) -> str:
    """Build a grounded prompt using a compact, single-paragraph instruction.

    Variant B for LLM evaluation: keeps the same safety and grounding
    constraints as ``build_prompt``, but drops the explanatory scaffolding
    and the worked citation example, to test whether a shorter prompt
    performs as well as the detailed one.
    """
    context = format_context(chunks)

    return f"""Contexto:
{context}

Pregunta: {question}

Respondé en español usando solo el contexto de arriba. Si no alcanza, decilo. \
No diagnostiques animales ni indiques dosis: sugerí consultar a un veterinario. \
Citá cada afirmación con [documento.pdf: página]."""


def extract_sources(chunks: list[dict[str, str | int | float]]) -> list[dict[str, str | int]]:
    """Return unique document-page pairs from retrieved chunks."""
    sources = []
    seen_sources = set()

    for chunk in chunks:
        source = (str(chunk["document"]), int(chunk["page"]))
        if source not in seen_sources:
            sources.append({"document": source[0], "page": source[1]})
            seen_sources.add(source)

    return sources


def answer_question(
    question: str,
    chunks: list[dict[str, str | int | float]],
    prompt_builder: Callable[[str, list[dict]], str] = build_prompt,
    model: str | None = None,
) -> dict[str, str | list[dict[str, str | int]]]:
    """Generate a grounded answer and return it with its retrieved sources.

    ``prompt_builder`` lets callers swap in a different prompt variant
    (e.g. ``build_prompt_concise``) without duplicating the generation
    logic, which is what the LLM evaluation script uses to compare them.
    ``model`` overrides the configured generation model (used by the
    evaluation, which records which model produced each answer).
    """
    if not chunks:
        raise ValueError("At least one retrieved chunk is required to answer a question")

    settings = get_settings()
    completion = complete(
        prompt_builder(question, chunks),
        model=model or settings.generation_model,
        temperature=settings.generation_temperature,
        max_tokens=settings.generation_max_tokens,
    )
    sources = extract_sources(chunks)

    if completion.finish_reason == "length":
        # The response was cut off before the model finished. Returning the
        # cut-off text silently would look like a complete (but wrong) answer.
        return {"answer": TRUNCATED_MESSAGE, "sources": sources}

    if not completion.text:
        return {"answer": NO_ANSWER_MESSAGE, "sources": sources}

    return {"answer": completion.text, "sources": sources}
