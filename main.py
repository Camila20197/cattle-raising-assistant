"""FastAPI application exposing the cattle-health RAG assistant.

Run locally with:
    uv run uvicorn main:app --reload

Then try it at http://127.0.0.1:8000/docs (interactive Swagger UI, generated
automatically by FastAPI from the models below).
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.generation.rag import answer_question
from src.retrieval.keyword_search import build_bm25_index, search_chunks

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
TOP_K = 5

# Holds the corpus and the BM25 index built once at startup, shared across
# every request. See build_bm25_index's docstring for why this matters.
app_state: dict = {}


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the corpus and build the BM25 index once, before serving traffic."""
    chunks = load_jsonl(CHUNKS_PATH)
    app_state["chunks"] = chunks
    app_state["bm25"] = build_bm25_index(chunks)
    yield
    app_state.clear()


app = FastAPI(
    title="Cattle Raising Assistant",
    description=(
        "RAG assistant that answers cattle-health questions grounded in "
        "INTA technical reports, with source citations."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, description="The producer's question, in Spanish."
    )


class Source(BaseModel):
    document: str
    page: int


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check. Deliberately does not call Gemini: it must stay fast
    and free, since it can be polled frequently by an orchestrator or load
    balancer once this is deployed."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Answer a cattle-health question grounded in the INTA knowledge base."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    retrieved = search_chunks(
        question, app_state["chunks"], top_k=TOP_K, bm25=app_state["bm25"]
    )

    if not retrieved:
        return AskResponse(
            answer="No encontré información relacionada en los documentos disponibles.",
            sources=[],
        )

    result = answer_question(question, retrieved)
    return AskResponse(answer=result["answer"], sources=result["sources"])