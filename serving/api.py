"""
GenAI Sandbox — FastAPI Serving Layer
======================================
Provides REST endpoints that delegate to whichever pipeline
template is configured in stack.yaml (naive_rag | agentic_rag).

Endpoints:
  GET  /            → health check + config summary
  POST /ingest      → index documents in /data
  POST /query       → {"question": str, "session_id"?: str}
  GET  /docs        → Swagger UI (auto-generated)
  GET  /metrics     → basic usage metrics
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

load_dotenv()

TEMPLATE = os.getenv("SANDBOX_TEMPLATE", os.getenv("TEMPLATE", "naive_rag"))

# ── Provider → required env-var mapping ───────────────────────
_PROVIDER_KEY_MAP: dict[str, str] = {
    "openai":       "OPENAI_API_KEY",
    "anthropic":    "ANTHROPIC_API_KEY",
    "google":       "GOOGLE_API_KEY",
    "groq":         "GROQ_API_KEY",
    "mistral":      "MISTRAL_API_KEY",
    "cohere":       "COHERE_API_KEY",
    "together":     "TOGETHER_API_KEY",
    "bedrock":      "AWS_ACCESS_KEY_ID",   # plus AWS_SECRET_ACCESS_KEY
    "azure-openai": "AZURE_OPENAI_API_KEY",
    # ollama: local — no key needed
}


def _check_api_key() -> str | None:
    """
    Returns a human-readable warning string if the configured provider's
    API key is missing or still contains a placeholder. Returns None if OK.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider == "ollama":
        return None  # no key needed

    env_var = _PROVIDER_KEY_MAP.get(provider)
    if not env_var:
        return None  # unknown provider — let pipeline fail with its own error

    value = os.getenv(env_var, "").strip()
    if not value or value.startswith("your-") or value in ("sk-", ""):
        return (
            f"\n"
            f"  ┌─────────────────────────────────────────────────────────┐\n"
            f"  │  ⚠  API key not configured                               │\n"
            f"  │                                                           │\n"
            f"  │  Provider : {provider:<46}│\n"
            f"  │  Env var  : {env_var:<46}│\n"
            f"  │                                                           │\n"
            f"  │  Run one of the following to set your key:               │\n"
            f"  │    python3 scripts/configure.py                          │\n"
            f"  │    make configure                                         │\n"
            f"  └─────────────────────────────────────────────────────────┘\n"
        )
    return None

# ── Lazy pipeline loader ───────────────────────────────────────
_pipeline: Any = None


def _load_pipeline() -> Any:
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    if TEMPLATE == "agentic_rag":
        from templates.agentic_rag.pipeline import get_pipeline
    else:
        from templates.naive_rag.pipeline import get_pipeline

    _pipeline = get_pipeline()
    return _pipeline


# ── Request / Response models ─────────────────────────────────
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to answer.")
    session_id: str = Field(default="default", description="Conversation thread ID (agentic_rag).")


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict] = []
    metadata: dict = {}


class IngestResponse(BaseModel):
    indexed: int
    source_files: int | None = None
    message: str | None = None


# ── Metrics (in-memory, replace with Prometheus in prod) ──────
_metrics: dict[str, int | float] = {
    "queries_total": 0,
    "ingest_total": 0,
    "errors_total": 0,
    "total_latency_ms": 0.0,
}


# ── App lifecycle ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"GenAI Sandbox starting — template: {TEMPLATE}")
    warning = _check_api_key()
    if warning:
        logger.warning(warning)
    else:
        logger.info("API key check passed.")
    logger.info("Pipeline loads lazily on first /ingest or /query call.")
    yield
    logger.info("Sandbox shutting down.")


# ── App initialization ─────────────────────────────────────────
app = FastAPI(
    title="GenAI Sandbox API",
    description=(
        "A dynamic RAG & Agentic AI sandbox. "
        "Configure your stack in /config/stack.yaml and ingest documents via /ingest."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Middleware: request timing ─────────────────────────────────
@app.middleware("http")
async def add_timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = str(round(elapsed, 2))
    return response


# ══════════════════════════════════════════════════════════════
#  Endpoints
# ══════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
async def health() -> dict:
    return {
        "status": "ok",
        "sandbox": "GenAI Sandbox Wrapper",
        "template": TEMPLATE,
        "orchestrator": os.getenv("SANDBOX_ORCHESTRATOR", "langchain"),
        "vector_db": os.getenv("SANDBOX_VECTOR_DB", "chroma"),
        "llm_provider": os.getenv("SANDBOX_LLM_PROVIDER", "openai"),
    }


@app.post("/ingest", response_model=IngestResponse, tags=["Data"])
async def ingest() -> IngestResponse:
    """
    Index all documents in the /data volume.

    Scans for PDF, CSV, Markdown, and TXT files, splits them into
    chunks, and upserts them into the configured vector store.
    Re-running ingest after adding new files is safe (upsert).
    """
    warning = _check_api_key()
    if warning:
        provider = os.getenv("LLM_PROVIDER", "openai")
        raise HTTPException(
            status_code=422,
            detail=f"API key for '{provider}' is not set. Run: python3 scripts/configure.py",
        )
    try:
        pipeline = _load_pipeline()
        result = pipeline.ingest()
        _metrics["ingest_total"] += 1
        return IngestResponse(**result)
    except Exception as exc:
        _metrics["errors_total"] += 1
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(req: QueryRequest) -> QueryResponse:
    """
    Answer a question using the indexed knowledge base.

    - **naive_rag**: single-pass retrieval + generation
    - **agentic_rag**: routed, graded, with fallback
    """
    warning = _check_api_key()
    if warning:
        provider = os.getenv("LLM_PROVIDER", "openai")
        raise HTTPException(
            status_code=422,
            detail=f"API key for '{provider}' is not set. Run: python3 scripts/configure.py",
        )
    try:
        t0 = time.perf_counter()
        pipeline = _load_pipeline()
        result = pipeline.query(req.question, session_id=req.session_id)
        elapsed = (time.perf_counter() - t0) * 1000
        _metrics["queries_total"] += 1
        _metrics["total_latency_ms"] += elapsed
        return QueryResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            metadata={
                "template": TEMPLATE,
                "latency_ms": round(elapsed, 2),
                "retrieval_used": result.get("retrieval_used"),
                "grade_passed": result.get("grade_passed"),
            },
        )
    except Exception as exc:
        _metrics["errors_total"] += 1
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/metrics", tags=["Observability"])
async def metrics() -> dict:
    q = _metrics["queries_total"]
    avg_latency = (_metrics["total_latency_ms"] / q) if q > 0 else 0
    return {
        **_metrics,
        "avg_latency_ms": round(avg_latency, 2),
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}")
    _metrics["errors_total"] += 1
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
