# ============================================================
#  GenAI Sandbox Wrapper — Dockerfile
#  Build context: genai-sandbox/  (the project root)
#  Supports: LangChain · LangGraph · LlamaIndex · CrewAI
#            OpenAI · Anthropic · Google · Ollama · Groq · Mistral
#            Chroma · Qdrant · PGVector · Pinecone · FAISS · Weaviate · Milvus
#            FastAPI · Streamlit · Chainlit · Jupyter
# ============================================================
FROM python:3.11-slim AS base

LABEL maintainer="genai-sandbox"
LABEL description="One-stop reproducible GenAI / RAG / Agentic-AI developer sandbox v2"
LABEL version="2.0.0"

# ── System dependencies ───────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libpq-dev \
        libmagic-dev \
        poppler-utils \
        tesseract-ocr \
        pandoc \
        ffmpeg \
        make \
    && rm -rf /var/lib/apt/lists/*

# ── Install uv (Astral — fastest Python package manager 2025) ─
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:/root/.local/bin:$PATH"

# ── Working directory ─────────────────────────────────────────
WORKDIR /app

# ── Copy application code (context = genai-sandbox/ root) ─────
COPY serving/          /app/serving/
COPY templates/        /app/templates/
COPY scripts/          /app/scripts/

# ── Entrypoint ────────────────────────────────────────────────
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# ── Pre-install always-needed base packages at build time ─────
#    Heavy framework packages installed AT RUNTIME by entrypoint
#    based on stack.yaml selection (via uv — takes < 30 seconds)
RUN uv pip install --system \
    "fastapi>=0.115" \
    "uvicorn[standard]>=0.30" \
    "python-dotenv>=1.0" \
    "pydantic>=2.7" \
    "pydantic-settings>=2.3" \
    "httpx>=0.27" \
    "loguru>=0.7" \
    "tenacity>=8.3" \
    "rich>=13.7" \
    "pyyaml>=6.0" \
    "requests>=2.32" \
    "aiohttp>=3.9" \
    "anyio>=4.4" \
    "structlog>=24.1"

# ── Volume mount points ───────────────────────────────────────
#   /app/data   → user RAG files (PDF, CSV, DOCX, MD, TXT, PPTX, HTML)
#   /app/src    → custom agent scripts (hot-reloaded)
#   /app/config → stack.yaml + env overrides
VOLUME ["/app/data", "/app/src", "/app/config"]

# ── Port map ──────────────────────────────────────────────────
#  8000 FastAPI  |  8501 Streamlit  |  8080 Chainlit  |  8888 Jupyter
EXPOSE 8000 8501 8080 8888

ENTRYPOINT ["/entrypoint.sh"]
