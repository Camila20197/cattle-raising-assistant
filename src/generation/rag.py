"""Prompt construction and Groq generation for the RAG application."""

from src.generation.groq_client import GENERATION_MODEL, get_groq_client

GENERATION_TEMPERATURE = 0.1
GENERATION_MAX_OUTPUT_TOKENS = 2_048
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
    prompt_builder=build_prompt,
) -> dict[str, str | list[dict[str, str | int]]]:
    """Generate a grounded answer and return it with its retrieved sources.

    ``prompt_builder`` lets callers swap in a different prompt variant
    (e.g. ``build_prompt_concise``) without duplicating the generation
    logic, which is what the LLM evaluation script uses to compare them.
    """
    if not chunks:
        raise ValueError("At least one retrieved chunk is required to answer a question")

    client = get_groq_client()
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt_builder(question, chunks)}],
        temperature=GENERATION_TEMPERATURE,
        max_tokens=GENERATION_MAX_OUTPUT_TOKENS,
    )

    choice = response.choices[0]
    text = choice.message.content

    if choice.finish_reason == "length":
        # Groq's (OpenAI-style) signal that the response was cut off before
        # the model finished -- same failure mode we hit with Gemini, same
        # reasoning for not silently returning the cut-off text.
        return {
            "answer": TRUNCATED_MESSAGE,
            "sources": extract_sources(chunks),
        }

    if not text:
        return {
            "answer": NO_ANSWER_MESSAGE,
            "sources": extract_sources(chunks),
        }

    return {
        "answer": text.strip(),
        "sources": extract_sources(chunks),
    }