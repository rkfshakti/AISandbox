#!/usr/bin/env bash
# ==============================================================
#  GenAI Sandbox Wrapper v2 — Container Entrypoint
#
#  Reads /app/config/stack.yaml, dynamically resolves
#  the full dependency graph for the requested stack, installs
#  everything in < 60s via uv, then launches the serving layer.
#
#  Supported Orchestrators : langchain | langgraph | llamaindex |
#                            crewai | autogen | dspy | pydantic-ai
#  Supported Vector DBs    : chroma | qdrant | pgvector | pinecone |
#                            weaviate | faiss | milvus | opensearch
#  Supported LLM Providers : openai | anthropic | google | ollama |
#                            groq | mistral | cohere | together |
#                            bedrock | azure-openai | nvidia
#  Supported Serving       : fastapi | streamlit | chainlit | jupyter
# ==============================================================
set -euo pipefail

CYAN='\033[0;36m'; GREEN='\033[0;32m'
YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${CYAN}[sandbox]${NC} $*"; }
ok()   { echo -e "${GREEN}[✓ sandbox]${NC} $*"; }
warn() { echo -e "${YELLOW}[! sandbox]${NC} $*"; }
err()  { echo -e "${RED}[✗ sandbox]${NC} $*"; exit 1; }

CONFIG_FILE="/app/config/stack.yaml"
DEFAULT_CONFIG="/app/scripts/default_stack.yaml"

# ═══════════════════════════════════════════════════════════════
#  STEP 1 — Resolve configuration
# ═══════════════════════════════════════════════════════════════
if [ ! -f "$CONFIG_FILE" ]; then
    warn "No /config/stack.yaml found — copying defaults."
    cp "$DEFAULT_CONFIG" "$CONFIG_FILE"
fi

parse_yaml() {
    local key="$1"
    grep -E "^${key}:" "$CONFIG_FILE" \
        | awk -F': ' '{print $2}' \
        | tr -d '"' | tr -d "'" | xargs
}

ORCHESTRATOR=$(parse_yaml "orchestrator");  ORCHESTRATOR=${ORCHESTRATOR:-langchain}
VECTOR_DB=$(parse_yaml "vector_db");        VECTOR_DB=${VECTOR_DB:-chroma}
SERVING=$(parse_yaml "serving");            SERVING=${SERVING:-fastapi}
TEMPLATE=$(parse_yaml "template");          TEMPLATE=${TEMPLATE:-naive_rag}
LLM_PROVIDER=$(parse_yaml "llm_provider"); LLM_PROVIDER=${LLM_PROVIDER:-openai}
CHAT_MODEL=$(parse_yaml "chat_model");      CHAT_MODEL=${CHAT_MODEL:-gpt-4o-mini}
EMBED_MODEL=$(parse_yaml "embedding_model"); EMBED_MODEL=${EMBED_MODEL:-text-embedding-3-small}
CHUNK_SIZE=$(parse_yaml "chunk_size");      CHUNK_SIZE=${CHUNK_SIZE:-1000}
CHUNK_OVERLAP=$(parse_yaml "chunk_overlap"); CHUNK_OVERLAP=${CHUNK_OVERLAP:-150}
TOP_K=$(parse_yaml "top_k");                TOP_K=${TOP_K:-5}
COLLECTION=$(parse_yaml "collection_name"); COLLECTION=${COLLECTION:-sandbox_docs}

log "╔══════════════════════════════════════════╗"
log "║    AISandbox — GenAI Stack Wrapper        ║"
log "║    v3.0.0  •  April 2026                  ║"
log "╠══════════════════════════════════════════╣"
log "║  Orchestrator : ${ORCHESTRATOR}"
log "║  Vector DB    : ${VECTOR_DB}"
log "║  LLM Provider : ${LLM_PROVIDER}"
log "║  Template     : ${TEMPLATE}"
log "║  Serving      : ${SERVING}"
log "╚══════════════════════════════════════════╝"

# ═══════════════════════════════════════════════════════════════
#  STEP 2 — Assemble dependency list
# ═══════════════════════════════════════════════════════════════
PKGS=(
    # ── Core runtime (always installed) ───────────────────────
    "fastapi>=0.135.3"
    "uvicorn[standard]>=0.44.0"
    "python-dotenv>=1.0"
    "pydantic>=2.12.5"
    "pydantic-settings>=2.13.1"
    "httpx>=0.28.1"
    "loguru>=0.7"
    "tenacity>=9.0"
    "rich>=13.9"
    "pyyaml>=6.0"
    "structlog>=25.1"
    "aiofiles>=24.1"
    "anyio>=4.9"
    # ── Document loaders (universal) ──────────────────────────
    "pypdf>=6.10.0"
    "pdfminer.six"
    "python-docx>=1.1"
    "python-pptx>=1.0"
    "openpyxl>=3.1"
    "pandas>=3.0.2"
    "beautifulsoup4>=4.12"
    "lxml>=5.3"
    "markdown"
    "python-magic"
    "tiktoken>=0.12.0"
    # ── Utilities ─────────────────────────────────────────────
    "numpy>=2.4.4"
    "scipy>=1.15"
    "tqdm>=4.67"
    "click>=8.1"
    "jinja2>=3.1"
)

# ── ORCHESTRATOR packages ─────────────────────────────────────
case "$ORCHESTRATOR" in
    langchain)
        PKGS+=(
            "langchain>=1.2.15"
            "langchain-core>=1.2.28"
            "langchain-community>=0.4.1"
            "langchain-text-splitters>=1.1.1"
            "langsmith>=0.7.30"
        )
        ;;
    langgraph)
        PKGS+=(
            "langchain>=1.2.15"
            "langchain-core>=1.2.28"
            "langchain-community>=0.4.1"
            "langchain-text-splitters>=1.1.1"
            "langgraph>=1.1.6"
            "langgraph-checkpoint>=4.0.1"
            "langgraph-sdk>=0.3.13"
            "langsmith>=0.7.30"
        )
        ;;
    llamaindex)
        PKGS+=(
            "llama-index>=0.14.20"
            "llama-index-core>=0.14.20"
            "llama-index-readers-file>=0.4"
            "llama-index-embeddings-openai>=0.3"
            "llama-index-llms-openai>=0.4"
            "llama-index-vector-stores-chroma>=0.4"
            "llama-index-postprocessor-flag-embedding-reranker>=0.3"
        )
        ;;
    crewai)
        PKGS+=(
            "crewai>=1.14.1"
            "crewai-tools>=0.35"
            "langchain>=1.2.15"
            "langchain-community>=0.4.1"
        )
        ;;
    autogen)
        PKGS+=(
            "autogen-agentchat>=0.7.5"
            "autogen-ext[openai,anthropic]>=0.7.5"
            "autogen-core>=0.7.5"
        )
        ;;
    dspy)
        PKGS+=(
            "dspy>=3.1.3"
        )
        ;;
    pydantic-ai)
        PKGS+=(
            "pydantic-ai>=1.80.0"
        )
        ;;
    *)
        warn "Unknown orchestrator '${ORCHESTRATOR}' — defaulting to langchain."
        PKGS+=(
            "langchain>=0.3"
            "langchain-core>=0.3"
            "langchain-community>=0.3"
            "langchain-text-splitters>=0.3"
            "langsmith>=0.1"
        )
        ;;
esac

# ── VECTOR DB packages ────────────────────────────────────────
case "$VECTOR_DB" in
    chroma)
        PKGS+=("chromadb>=1.5.7" "langchain-chroma>=1.1.0")
        ;;
    qdrant)
        PKGS+=("qdrant-client>=1.17.1" "langchain-qdrant>=1.1.0")
        ;;
    pgvector)
        PKGS+=("psycopg2-binary" "pgvector>=0.4.2" "langchain-postgres>=0.0.17")
        ;;
    pinecone)
        PKGS+=("pinecone>=8.1.2" "langchain-pinecone>=0.2.13")
        ;;
    weaviate)
        PKGS+=("weaviate-client>=4.20.5" "langchain-weaviate>=0.0.6")
        ;;
    faiss)
        PKGS+=("faiss-cpu>=1.13.2" "langchain-community>=0.4.1")
        ;;
    milvus)
        PKGS+=("pymilvus>=2.6.12" "langchain-milvus>=0.3.3")
        ;;
    opensearch)
        PKGS+=("opensearch-py>=2.8.0" "langchain-community>=0.4.1")
        ;;
    *)
        warn "Unknown vector_db '${VECTOR_DB}' — defaulting to chroma."
        PKGS+=("chromadb>=0.5" "langchain-chroma>=0.1")
        ;;
esac

# ── LLM PROVIDER packages ─────────────────────────────────────
case "$LLM_PROVIDER" in
    openai)
        PKGS+=("openai>=2.31.0" "langchain-openai>=1.1.12")
        ;;
    anthropic)
        PKGS+=("anthropic>=0.94.0" "langchain-anthropic>=1.4.0")
        ;;
    google)
        PKGS+=(
            "google-genai>=1.72.0"
            "langchain-google-genai>=4.2.1"
            "google-cloud-aiplatform>=1.89.0"
            "langchain-google-vertexai>=3.2.2"
        )
        ;;
    ollama)
        PKGS+=("ollama>=0.6.1" "langchain-ollama>=1.1.0")
        ;;
    groq)
        PKGS+=("groq>=1.1.2" "langchain-groq>=1.1.2")
        ;;
    mistral)
        PKGS+=("mistralai>=2.3.2" "langchain-mistralai>=1.1.2")
        ;;
    cohere)
        PKGS+=("cohere>=6.1.0" "langchain-cohere>=0.5.0")
        ;;
    together)
        PKGS+=("together>=2.7.0" "langchain-together>=0.4.0")
        ;;
    azure-openai)
        PKGS+=("openai>=2.31.0" "langchain-openai>=1.1.12")
        ;;
    bedrock)
        PKGS+=("boto3>=1.42.88" "langchain-aws>=1.4.3")
        ;;
    nvidia)
        PKGS+=("langchain-nvidia-ai-endpoints>=1.2.1" "openai>=2.31.0")
        ;;
    *)
        warn "Unknown llm_provider '${LLM_PROVIDER}' — defaulting to openai."
        PKGS+=("openai>=1.35" "langchain-openai>=0.2")
        ;;
esac

# ── SERVING layer packages ────────────────────────────────────
case "$SERVING" in
    streamlit)
        PKGS+=("streamlit>=1.56.0" "watchdog>=6.0")
        ;;
    chainlit)
        PKGS+=("chainlit>=2.11.0")
        ;;
    jupyter)
        PKGS+=(
            "jupyterlab>=4.5.6"
            "ipywidgets>=8.1"
            "ipython>=8.36"
            "nbformat>=5.10"
        )
        ;;
    fastapi) ;; # already in base
    *)
        warn "Unknown serving '${SERVING}' — using fastapi."
        ;;
esac

# ── OBSERVABILITY (always on) ─────────────────────────────────
PKGS+=(
    "langsmith>=0.7.30"
    "opentelemetry-api>=1.30.0"
    "opentelemetry-sdk>=1.30.0"
    "prometheus-client>=0.22.0"
)

# ── RERANKING / ADVANCED RETRIEVAL ───────────────────────────
PKGS+=(
    "sentence-transformers>=5.4.0"
    "rank-bm25>=0.2"
    "flashrank>=0.2"
)

    # ── UNSTRUCTURED (multi-format document parsing) ──────────────
PKGS+=(
    "unstructured[pdf,docx,pptx,xlsx,md,html]>=0.22.18"
)

# ═══════════════════════════════════════════════════════════════
#  STEP 3 — Install all packages via uv (lightning fast)
# ═══════════════════════════════════════════════════════════════
log "Installing ${#PKGS[@]} packages via uv  ⚡ …"
uv pip install --system --no-progress "${PKGS[@]}" 2>&1 \
    | grep -E "(Installed|Updated|Resolved|error|Error)" || true
ok "All dependencies installed."

# ═══════════════════════════════════════════════════════════════
#  STEP 4 — Bootstrap template into /src (first boot only)
# ═══════════════════════════════════════════════════════════════
TEMPLATE_DIR="/app/templates/${TEMPLATE}"
SRC_DIR="/app/src"

SRC_FILE_COUNT=$(find "${SRC_DIR}" -maxdepth 1 -name "*.py" 2>/dev/null | wc -l)
if [ "${SRC_FILE_COUNT}" -eq 0 ]; then
    if [ -d "$TEMPLATE_DIR" ]; then
        log "First boot — bootstrapping '${TEMPLATE}' into /src …"
        cp -r "${TEMPLATE_DIR}/." "${SRC_DIR}/"
        ok "Template loaded. Edit files in /src, then restart to pick up changes."
    else
        err "Template '${TEMPLATE}' not found at ${TEMPLATE_DIR}."
    fi
else
    log "Custom files found in /src — skipping template bootstrap."
fi

# ═══════════════════════════════════════════════════════════════
#  STEP 5 — Export env vars for use by child processes
# ═══════════════════════════════════════════════════════════════
export SANDBOX_ORCHESTRATOR="$ORCHESTRATOR"
export SANDBOX_VECTOR_DB="$VECTOR_DB"
export SANDBOX_LLM_PROVIDER="$LLM_PROVIDER"
export SANDBOX_TEMPLATE="$TEMPLATE"
export TEMPLATE="$TEMPLATE"
export CHAT_MODEL="$CHAT_MODEL"
export EMBEDDING_MODEL="$EMBED_MODEL"
export CHUNK_SIZE="$CHUNK_SIZE"
export CHUNK_OVERLAP="$CHUNK_OVERLAP"
export TOP_K="$TOP_K"
export COLLECTION_NAME="$COLLECTION"

# ═══════════════════════════════════════════════════════════════
#  STEP 6 — Launch serving layer
# ═══════════════════════════════════════════════════════════════
ok "🚀 Launching serving layer: [${SERVING}]"

case "$SERVING" in
    fastapi)
        exec uvicorn serving.api:app \
            --host 0.0.0.0 --port 8000 \
            --reload \
            --reload-dir /app/src \
            --reload-dir /app/serving \
            --log-level info
        ;;
    streamlit)
        exec streamlit run /app/serving/ui_streamlit.py \
            --server.port 8501 \
            --server.address 0.0.0.0 \
            --server.headless true \
            --server.fileWatcherType poll
        ;;
    chainlit)
        exec chainlit run /app/serving/ui_chainlit.py \
            --host 0.0.0.0 \
            --port 8080 \
            --watch
        ;;
    jupyter)
        exec jupyter lab \
            --ip=0.0.0.0 \
            --port=8888 \
            --no-browser \
            --allow-root \
            --notebook-dir=/app \
            --LabApp.token='' \
            --LabApp.password=''
        ;;
    *)
        err "Unknown serving='${SERVING}' in stack.yaml. Valid: fastapi | streamlit | chainlit | jupyter"
        ;;
esac
