import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from src.api.quota import QuestionQuota, QuotaExceededError
from src.config import get_settings
from src.generation.groq_client import LLMError, ModelUnavailableError
from src.generation.rag import answer_question
from src.jsonl import read_jsonl
from src.monitoring.db import init_db, log_conversation, log_feedback, monitoring_enabled
from src.retrieval.keyword_search import build_bm25_index, search_chunks

NO_DOCUMENTS_MESSAGE = "No encontré información relacionada en los documentos disponibles."

logger = logging.getLogger("cattle_assistant")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Holds the corpus and the BM25 index built once at startup, shared across
# every request. See build_bm25_index's docstring for why this matters.
app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the corpus, build the BM25 index, and prepare the monitoring
    tables once, before serving traffic."""
    settings = get_settings()
    chunks = read_jsonl(settings.chunks_path)
    app_state["chunks"] = chunks
    app_state["bm25"] = build_bm25_index(chunks)
    app_state["quota"] = QuestionQuota(
        per_minute=settings.max_questions_per_minute, per_day=settings.max_questions_per_day
    )
    init_db()
    logger.info(
        "Loaded %s chunks; generation model: %s", len(chunks), settings.generation_model
    )
    yield
    app_state.clear()


app = FastAPI(
    title="Cattle Raising Assistant",
    description=(
        "RAG assistant that answers cattle-health questions grounded in "
        "INTA technical reports, with source citations."
    ),
    version="0.2.0",
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
    conversation_id: int | None = Field(
        description="Id to send feedback about this answer; null when monitoring is disabled."
    )
    answer: str
    sources: list[Source]
    model: str = Field(description="LLM that generated the answer.")


class FeedbackRequest(BaseModel):
    conversation_id: int
    rating: Literal["up", "down"]
    comment: str | None = None


@app.exception_handler(ModelUnavailableError)
def model_unavailable_handler(request: Request, error: ModelUnavailableError) -> JSONResponse:
    """A retired/unknown model is a configuration problem, not a server crash:
    answer 503 with a message that says exactly what to change, instead of an
    opaque 500."""
    logger.error("%s", error)
    return JSONResponse(status_code=503, content={"detail": str(error)})


@app.exception_handler(QuotaExceededError)
def quota_exceeded_handler(request: Request, error: QuotaExceededError) -> JSONResponse:
    """Cost safeguard: too many questions overall. Answer 429 without calling the LLM."""
    logger.warning("%s", error)
    return JSONResponse(
        status_code=429,
        content={"detail": str(error)},
        headers={"Retry-After": str(error.retry_after_seconds)},
    )


@app.exception_handler(LLMError)
def llm_error_handler(request: Request, error: LLMError) -> JSONResponse:
    logger.error("LLM provider error: %s", error)
    return JSONResponse(
        status_code=503,
        content={"detail": "The language model is temporarily unavailable. Please try again."},
    )


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Send anyone who lands on the bare URL straight to the interactive docs,
    instead of a bare 404 -- convenient for humans, and one less thing to
    explain in the README."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check. Deliberately does not call the LLM: it must stay fast
    and free, since it can be polled frequently by an orchestrator or load
    balancer once this is deployed. It reports the configured model so a
    stale configuration is easy to spot."""
    return {
        "status": "ok",
        "generation_model": get_settings().generation_model,
        "monitoring": "enabled" if monitoring_enabled() else "disabled",
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Answer a cattle-health question grounded in the INTA knowledge base."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    settings = get_settings()
    retrieved = search_chunks(
        question, app_state["chunks"], top_k=settings.top_k, bm25=app_state["bm25"]
    )

    if not retrieved:
        conversation_id = log_conversation(
            question, NO_DOCUMENTS_MESSAGE, sources=[], generation_model="none"
        )
        return AskResponse(
            conversation_id=conversation_id,
            answer=NO_DOCUMENTS_MESSAGE,
            sources=[],
            model="none",
        )

    # Only questions that will actually call the LLM count against the quota.
    app_state["quota"].acquire()
    result = answer_question(question, retrieved, model=settings.generation_model)
    conversation_id = log_conversation(
        question,
        result["answer"],
        sources=result["sources"],
        generation_model=settings.generation_model,
    )
    logger.info("Answered conversation %s with %s sources", conversation_id, len(result["sources"]))

    return AskResponse(
        conversation_id=conversation_id,
        answer=result["answer"],
        sources=result["sources"],
        model=settings.generation_model,
    )


@app.post("/feedback")
def feedback(request: FeedbackRequest) -> dict[str, str]:
    """Record a thumbs up/down (and optional comment) for a previous /ask response."""
    if not monitoring_enabled():
        raise HTTPException(status_code=503, detail="Monitoring is disabled: DATABASE_URL is not set.")
    log_feedback(request.conversation_id, request.rating, request.comment)
    return {"status": "ok"}