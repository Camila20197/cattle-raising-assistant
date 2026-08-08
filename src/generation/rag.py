"""Prompt construction and Gemini generation for the RAG application."""

from google.genai import types

from src.retrieval.vector_search import get_client

GENERATION_MODEL = "gemini-3.5-flash"


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
- Cite every factual claim with its source in this format: [Document: page].

Source context:
{context}

User question: {question}
"""


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
    question: str, chunks: list[dict[str, str | int | float]]
) -> dict[str, str | list[dict[str, str | int]]]:
    """Generate a grounded answer and return it with its retrieved sources."""
    if not chunks:
        raise ValueError("At least one retrieved chunk is required to answer a question")

    client = get_client()
    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=build_prompt(question, chunks),
        config=types.GenerateContentConfig(
            max_output_tokens=1_000,
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        ),
    )

    return {
        "answer": response.text.strip(),
        "sources": extract_sources(chunks),
    }
