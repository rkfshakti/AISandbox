"""
GenAI Sandbox — FastAPI Serving Layer
======================================
Provides REST endpoints that delegate to whichever pipeline
template is configured in stack.yaml (naive_rag | agentic_rag).

Endpoints:
  GET  /            → health check + config summary
  POST /ingest      → index documents in /data
  POST /query       → {"question": str, "session_id"?: str}
  POST /stream      → Server-Sent Events streaming response
  GET  /docs        → Swagger UI (auto-generated)
  GET  /metrics     → usage metrics with token counts + cost
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

# allow `from src.llm_factory import ...` when running locally
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    elif TEMPLATE == "structured_output":
        from templates.structured_output.pipeline import get_pipeline
    elif TEMPLATE == "multi_agent":
        from templates.multi_agent.pipeline import get_pipeline
    else:
        from templates.naive_rag.pipeline import get_pipeline

    _pipeline = get_pipeline()
    return _pipeline


# ── Request / Response models ─────────────────────────────────
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to answer.")
    session_id: str = Field(default="default", description="Conversation thread ID (agentic_rag / multi_agent).")
    output_schema: str = Field(default="", description="structured_output template only: qa | summary | entities")


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
    "queries_total":      0,
    "streams_total":      0,
    "ingest_total":       0,
    "errors_total":       0,
    "total_latency_ms":   0.0,
    "total_input_tokens": 0,
    "total_output_tokens": 0,
    "total_cost_usd":     0.0,
}


def _record_token_usage(result: dict) -> None:
    """Extract token usage from pipeline result and update cost metrics."""
    try:
        from src.llm_factory import estimate_cost
        model         = os.getenv("CHAT_MODEL", "gpt-4o-mini")
        input_tokens  = result.get("input_tokens", 0)
        output_tokens = result.get("output_tokens", 0)
        if input_tokens or output_tokens:
            _metrics["total_input_tokens"]  += input_tokens
            _metrics["total_output_tokens"] += output_tokens
            _metrics["total_cost_usd"]      += estimate_cost(model, input_tokens, output_tokens)
    except Exception:
        pass  # cost tracking is best-effort


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
        # structured_output pipeline accepts output_schema kwarg
        if TEMPLATE == "structured_output" and req.output_schema:
            result = pipeline.query(req.question, session_id=req.session_id, output_schema=req.output_schema)
        else:
            result = pipeline.query(req.question, session_id=req.session_id)
        elapsed = (time.perf_counter() - t0) * 1000
        _metrics["queries_total"] += 1
        _metrics["total_latency_ms"] += elapsed
        _record_token_usage(result)
        return QueryResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            metadata={
                "template":      TEMPLATE,
                "latency_ms":    round(elapsed, 2),
                "retrieval_used": result.get("retrieval_used"),
                "grade_passed":   result.get("grade_passed"),
                "input_tokens":   result.get("input_tokens"),
                "output_tokens":  result.get("output_tokens"),
            },
        )
    except Exception as exc:
        _metrics["errors_total"] += 1
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/stream", tags=["Query"])
async def stream(req: QueryRequest) -> StreamingResponse:
    """
    Stream the answer token-by-token using Server-Sent Events (SSE).

    Connect with EventSource or curl:
        curl -N -X POST http://localhost:8000/stream \
             -H 'Content-Type: application/json' \
             -d '{"question": "What is RAG?"}'

    Each event is a JSON object: {"token": "..."}  
    A final event {"done": true, "sources": [...]} signals completion.
    """
    warning = _check_api_key()
    if warning:
        provider = os.getenv("LLM_PROVIDER", "openai")
        raise HTTPException(
            status_code=422,
            detail=f"API key for '{provider}' is not set. Run: python3 scripts/configure.py",
        )

    async def event_generator():
        try:
            pipeline = _load_pipeline()
            t0 = time.perf_counter()

            # naive_rag exposes an astream method; fall back to sync query
            if hasattr(pipeline, "astream"):
                sources = []
                async for chunk in pipeline.astream(req.question):
                    if isinstance(chunk, dict):
                        # final chunk carries sources
                        sources = chunk.get("sources", [])
                        token   = chunk.get("token", "")
                    else:
                        token = str(chunk)
                    if token:
                        yield f"data: {json.dumps({'token': token})}\n\n"
                elapsed = (time.perf_counter() - t0) * 1000
                _metrics["streams_total"] += 1
                _metrics["total_latency_ms"] += elapsed
                yield f"data: {json.dumps({'done': True, 'sources': sources, 'latency_ms': round(elapsed, 2)})}\n\n"
            else:
                # Fallback: run sync query, stream the answer word-by-word
                result  = pipeline.query(req.question, session_id=req.session_id)
                answer  = result.get("answer", "")
                sources = result.get("sources", [])
                words   = answer.split(" ")
                for i, word in enumerate(words):
                    token = word if i == len(words) - 1 else word + " "
                    yield f"data: {json.dumps({'token': token})}\n\n"
                elapsed = (time.perf_counter() - t0) * 1000
                _metrics["streams_total"] += 1
                _metrics["total_latency_ms"] += elapsed
                _record_token_usage(result)
                yield f"data: {json.dumps({'done': True, 'sources': sources, 'latency_ms': round(elapsed, 2)})}\n\n"
        except Exception as exc:
            _metrics["errors_total"] += 1
            logger.exception("Stream failed")
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/metrics", tags=["Observability"])
async def metrics() -> dict:
    q           = _metrics["queries_total"]
    s           = _metrics["streams_total"]
    total_reqs  = q + s
    avg_latency = (_metrics["total_latency_ms"] / total_reqs) if total_reqs > 0 else 0
    return {
        **_metrics,
        "avg_latency_ms":   round(avg_latency, 2),
        "total_cost_usd":   round(float(_metrics["total_cost_usd"]), 6),
        "cost_per_query":   round(float(_metrics["total_cost_usd"]) / q, 6) if q > 0 else 0.0,
        "llm_provider":     os.getenv("LLM_PROVIDER", "openai"),
        "chat_model":       os.getenv("CHAT_MODEL", "gpt-4o-mini"),
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}")
    _metrics["errors_total"] += 1
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
