import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from src.generation.rag import answer_question
from src.monitoring.db import init_db, log_conversation, log_feedback
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
    """Load the corpus, build the BM25 index, and prepare the monitoring
    tables once, before serving traffic."""
    chunks = load_jsonl(CHUNKS_PATH)
    app_state["chunks"] = chunks
    app_state["bm25"] = build_bm25_index(chunks)
    init_db()
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
    conversation_id: int
    answer: str
    sources: list[Source]


class FeedbackRequest(BaseModel):
    conversation_id: int
    rating: Literal["up", "down"]
    comment: str | None = None


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Send anyone who lands on the bare URL straight to the interactive docs,
    instead of a bare 404 -- convenient for humans, and one less thing to
    explain in the README."""
    return RedirectResponse(url="/docs")


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
        answer = "No encontré información relacionada en los documentos disponibles."
        conversation_id = log_conversation(question, answer, sources=[])
        return AskResponse(conversation_id=conversation_id, answer=answer, sources=[])

    result = answer_question(question, retrieved)
    conversation_id = log_conversation(question, result["answer"], sources=result["sources"])

    return AskResponse(
        conversation_id=conversation_id, answer=result["answer"], sources=result["sources"]
    )


@app.post("/feedback")
def feedback(request: FeedbackRequest) -> dict[str, str]:
    """Record a thumbs up/down (and optional comment) for a previous /ask response."""
    log_feedback(request.conversation_id, request.rating, request.comment)
    return {"status": "ok"}