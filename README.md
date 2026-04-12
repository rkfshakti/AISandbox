# AISandbox

> **Install once. Use every AI stack. Zero tension.**

AISandbox is a `create-react-app`-style CLI for RAG and Agentic AI workflows.  
Drop your documents in a folder, run `make configure`, and in minutes you have a fully working AI pipeline — with your choice of LLM provider, vector database, orchestration framework, and serving layer.

No package conflicts. No boilerplate. No `.env` hunting.

---

## Why AISandbox?

| Without AISandbox | With AISandbox |
|---|---|
| Spend hours resolving LangChain ↔ LangGraph version conflicts | `pip install aisandbox` — done |
| Manually wire ChromaDB, OpenAI, FastAPI together | Pick your stack in a YAML file |
| Forget which env var OpenAI vs Anthropic needs | `make configure` — interactive wizard asks you |
| Start from scratch every project | `aisandbox new my-project` — ready in 60 seconds |

---

## Quick Start (No Docker Needed)

```bash
# 1. Install
pip install aisandbox

# 2. Scaffold a project
aisandbox new my-rag-app
cd my-rag-app

# 3. Configure your LLM provider interactively (no manual .env editing)
make configure

# 4. Drop documents in ./data
cp ~/Downloads/*.pdf data/
cp ~/notes/*.md data/

# 5. Start the API
make serve

# 6. Index your documents
make ingest

# 7. Ask a question
make query Q="What is the main topic of these documents?"
```

Or with Docker:
```bash
make up       # docker compose up --build
make ingest
make query Q="summarize the key findings"
```

---

## Interactive Setup (`make configure`)

No need to read docs or find API key names. The wizard handles it:

```
╔══════════════════════════════════════════════════╗
║     AISandbox — Setup Wizard                     ║
╚══════════════════════════════════════════════════╝

── Step 1 of 4  —  Choose your LLM Provider ────────────

 ► 1) OpenAI          (gpt-5.4, gpt-5.4-mini, gpt-5.4-nano)
   2) Anthropic       (claude-opus-4-6, claude-sonnet-4-6)
   3) Google          (gemini-2.5-pro, gemini-2.5-flash)
   4) Groq            (Llama 3.3 70B — ultra-fast)
   5) Mistral AI      (Mistral Large, Mistral Small)
   6) Ollama          (Local LLMs — Llama 3, Qwen 2.5)
   7) Cohere          (Command R+)
   8) Together AI     (100+ open-source models)
   9) AWS Bedrock     (Claude, Titan, Llama via AWS)
  10) Azure OpenAI    (gpt-5.4 via Azure)

── Step 2 of 4  —  Enter your API Key ──────────────────

  Enter your OpenAI API key: **********************

── Step 3 of 4  —  Choose your Stack ───────────────────
  ...

✓ Configuration saved!
```

Writes `.env` and `config/stack.yaml` automatically. Key is never stored in shell history.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              aisandbox new my-project                    │
│                    (CLI Scaffold)                        │
└────────────────────────┬────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │    make configure   │  ← interactive API key wizard
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  docker-compose /   │
              │  make serve (local) │
              └──────────┬──────────┘
                         │
         ┌───────────────▼───────────────────┐
         │          entrypoint.sh             │
         │  reads stack.yaml → uv install     │
         │  bootstraps template → launches    │
         └───┬──────────┬──────────┬──────────┘
             │          │          │
      ┌──────▼──┐ ┌─────▼──┐ ┌────▼─────┐
      │ FastAPI  │ │Chainlit│ │Streamlit │
      │  :8000   │ │ :8080  │ │  :8501   │
      └──────┬───┘ └────┬───┘ └────┬─────┘
             └──────────┼──────────┘
                   ┌────▼────┐
                   │Pipeline │
                   │naive_rag│
                   │   or    │
                   │agentic_ │
                   │   rag   │
                   └────┬────┘
              ┌─────────▼──────────────────┐
              │        Vector Store          │
              │  Chroma · Qdrant · PGVector  │
              │  Pinecone · FAISS · Weaviate │
              │  Milvus · OpenSearch         │
              └─────────────────────────────┘
```

---

## Supported Stack (April 2026)

### Orchestrators
| Package | Version | Notes |
|---|---|---|
| **LangGraph** | `1.1.6` | Recommended — agentic loops, persistence |
| **LangChain** | `1.2.15` | Chains, LCEL, tool use |
| **LlamaIndex** | `0.14.20` | Advanced indexing, query engines |
| **CrewAI** | `1.14.1` | Multi-agent crews |
| **AutoGen** | `0.7.5` | Conversational multi-agent |
| **DSPy** | `3.1.3` | Declarative LM programs |
| **PydanticAI** | `1.80.0` | Structured, type-safe agents |

### LLM Providers
| Provider | Package | Model Examples |
|---|---|---|
| **OpenAI** | `openai 2.31` | gpt-5.4, gpt-5.4-mini, gpt-5.4-nano |
| **Anthropic** | `anthropic 0.94` | claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5 |
| **Google** | `google-genai 1.72` | gemini-2.5-pro, gemini-2.5-flash |
| **Groq** | `groq 1.1.2` | llama-3.3-70b (300 tok/s) |
| **Mistral** | `mistralai 2.3.2` | mistral-large-latest |
| **Ollama** | `ollama 0.6.1` | llama3.2, qwen2.5, phi-4 (local) |
| **Cohere** | `cohere 6.1` | command-r-plus |
| **Together AI** | `together 2.7` | 100+ open models |
| **AWS Bedrock** | `boto3 1.42` | claude, titan, llama via AWS |
| **Azure OpenAI** | `openai 2.31` | gpt-5.4 via Azure endpoints |

### Vector Databases
| DB | Package | Notes |
|---|---|---|
| **ChromaDB** | `1.5.7` | Zero-config, local (default) |
| **Qdrant** | `1.17.1` | High-performance, local or cloud |
| **FAISS** | `1.13.2` | In-memory, no side-car |
| **PGVector** | `0.4.2` | Postgres extension |
| **Pinecone** | `8.1.2` | Serverless cloud |
| **Weaviate** | `4.20.5` | Multi-modal, cloud |
| **Milvus** | `2.6.12` | Enterprise-grade |
| **OpenSearch** | `2.8.0` | AWS OpenSearch compatible |

### Serving Layers
| Layer | Port | Package |
|---|---|---|
| **FastAPI** | `:8000` | REST API + Swagger UI at `/docs` |
| **Chainlit** | `:8080` | Chat UI with conversation history — `2.11.0` |
| **Streamlit** | `:8501` | Rapid-prototype chat UI — `1.56.0` |
| **JupyterLab** | `:8888` | Notebook-first exploration — `4.5.6` |

---

## Configuration (`config/stack.yaml`)

```yaml
# Generated by: make configure
# Edit manually or re-run: python3 scripts/configure.py

orchestrator: langgraph        # langchain | langgraph | llamaindex | crewai | autogen | dspy | pydantic-ai
vector_db: chroma              # chroma | qdrant | pgvector | pinecone | faiss | weaviate | milvus | opensearch
llm_provider: openai           # openai | anthropic | google | groq | mistral | ollama | cohere | together | bedrock | azure-openai
template: naive_rag            # naive_rag | agentic_rag
serving: fastapi               # fastapi | chainlit | streamlit | jupyter

chat_model: gpt-5.4-mini
embedding_model: text-embedding-3-small
temperature: 0.0
max_tokens: 2048
chunk_size: 1000
chunk_overlap: 150
top_k: 5
collection_name: sandbox_docs
```

---

## Templates

### Template A — Naive RAG (`naive_rag`)

```
[Question] → [Embed] → [MMR Vector Search] → [Top-K Chunks]
                                                    │
                                          [LLM + System Prompt]
                                                    │
                                           [Answer + Sources]
```

Best for: Quick prototyping, single-domain Q&A, demos.

### Template B — Agentic RAG (`agentic_rag`)

```
[Question]
     │
[Router Node] ──── needs retrieval? ──No──► [Direct Generate]
     │Yes
[Retrieve Node] → [Grade Node] ──── pass? ──No──► [Fallback/Web]
                        │Yes
                  [Generate Node]
                        │
                 [Answer + Sources]
```

Best for: Complex multi-hop queries, production agents, fallback handling.

---

## REST API

When `serving: fastapi` — Swagger UI at [http://localhost:8000/docs](http://localhost:8000/docs)

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check + active config |
| `POST` | `/ingest` | Index all documents in `./data` |
| `POST` | `/query` | `{"question": "...", "session_id": "..."}` |
| `GET` | `/metrics` | Requests, latency, error counts |
| `GET` | `/docs` | Swagger UI |

---

## Makefile Commands

```bash
make configure    # interactive setup — choose provider & enter API key
make serve        # run API server locally (no Docker)
make up           # docker compose up --build
make down         # docker compose down
make ingest       # POST /ingest — index ./data documents
make query Q='…'  # POST /query — ask a question
make metrics      # GET /metrics
make logs         # stream container logs
make docs         # open Swagger UI in browser
make clean        # remove containers, volumes, cache
```

---

## CLI Reference

```bash
aisandbox new <name>                        # scaffold a new project
aisandbox new my-agent \
  --template agentic_rag \
  --vector-db qdrant \
  --serving chainlit \
  --llm anthropic
```

---

## Project Structure

```
my-project/
├── config/
│   └── stack.yaml          ← your stack configuration
├── data/                   ← drop PDFs, CSVs, Markdown, DOCX, HTML here
├── scripts/
│   └── configure.py        ← interactive setup wizard
├── src/                    ← your custom code (optional)
├── docker/
│   ├── Dockerfile
│   └── entrypoint.sh       ← auto-installs exact deps from stack.yaml
├── serving/
│   ├── api.py              ← FastAPI app
│   ├── ui_chainlit.py
│   └── ui_streamlit.py
├── templates/
│   ├── naive_rag/
│   │   └── pipeline.py
│   └── agentic_rag/
│       └── pipeline.py
├── cli/
│   └── scaffold.py         ← aisandbox new ... CLI
├── docker-compose.yaml
├── Makefile
├── pyproject.toml
└── .env.example
```

---

## Docker Profiles

```bash
make up                          # Sandbox + ChromaDB (default)
make up VECTOR_DB=qdrant         # + Qdrant
make up VECTOR_DB=pgvector       # + Postgres/pgvector
make up VECTOR_DB=milvus         # + Milvus
```

---

## License

MIT — use freely for personal and commercial projects.

---

*Built with Python 3.11+ · Powered by LangChain, LangGraph, LlamaIndex, and the full 2026 AI ecosystem.*


---

## What It Does

Drop your documents in a folder, choose your stack in a YAML file, and spin up a fully configured AI environment in seconds. No more fighting package conflicts, vector database setup, or boilerplate wiring.

```
genai-sandbox new my-project --template agentic_rag --vector-db qdrant --serving chainlit
cd my-project
cp .env.example .env        # add your API keys
make up                     # docker compose up --build
make ingest                 # index your documents
make query                  # ask a question
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    genai-sandbox new my-project                  │
│                          (CLI Scaffold)                          │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   docker-compose     │
                    │  (sandbox + DBs)     │
                    └──────────┬──────────┘
                               │
              ┌────────────────▼────────────────┐
              │       entrypoint.sh              │
              │  reads stack.yaml → uv install   │
              │  bootstraps template → launches  │
              └──┬─────────┬──────────┬─────────┘
                 │         │          │
          ┌──────▼──┐ ┌───▼────┐ ┌───▼──────┐
          │ FastAPI  │ │Chainlit│ │Streamlit │
          │  :8000   │ │ :8080  │ │  :8501   │
          └──────┬───┘ └───┬────┘ └───┬──────┘
                 └─────────┼──────────┘
                      ┌────▼────┐
                      │Pipeline │
                      │ naive_  │
                      │ rag /   │
                      │agentic_ │
                      │  rag    │
                      └────┬────┘
                      ┌────▼────────────────────┐
                      │     Vector Store          │
                      │  Chroma / Qdrant /        │
                      │  PGVector / Pinecone /    │
                      │  FAISS / Weaviate         │
                      └──────────────────────────┘
```

---

## Quick Start

### 1. Install the CLI

```bash
pip install genai-sandbox
# or from source:
pip install -e .
```

### 2. Scaffold a new project

```bash
# Simple RAG with FastAPI + ChromaDB + OpenAI
genai-sandbox new my-rag-app

# Advanced: Agentic RAG with Chainlit UI + Qdrant + Anthropic
genai-sandbox new my-agent \
  --template agentic_rag \
  --vector-db qdrant \
  --serving chainlit \
  --llm anthropic
```

### 3. Configure

```bash
cd my-rag-app
cp .env.example .env       # fill in your API keys
# Optionally edit config/stack.yaml for fine-grained settings
```

### 4. Add documents

```bash
cp ~/Downloads/*.pdf data/
cp ~/notes/*.md data/
```

### 5. Start

```bash
make up        # spins up Docker containers
make ingest    # indexes /data into the vector store
make query     # interactive Q&A via CLI
```

---

## Configuration (`config/stack.yaml`)

```yaml
orchestrator: langgraph       # langchain | langgraph | llamaindex
vector_db: chroma             # chroma | qdrant | pgvector | pinecone | faiss
llm_provider: openai          # openai | anthropic | google | ollama | groq
template: agentic_rag         # naive_rag | agentic_rag
serving: fastapi              # fastapi | streamlit | chainlit | jupyter

chat_model: gpt-5.4-mini
embedding_model: text-embedding-3-small
temperature: 0.0
chunk_size: 1000
chunk_overlap: 150
top_k: 5
```

Restart the container after changing this file.

---

## Templates

### Template A — Naive RAG (`template: naive_rag`)

```
[User Question]
      │
      ▼
[Embed Question] ──► [Vector Search (MMR)] ──► [Top-K Chunks]
                                                      │
                                              [LLM + System Prompt]
                                                      │
                                                [Answer + Sources]
```

**Best for:** Quick prototyping, single-domain Q&A, demos.

### Template B — Agentic RAG (`template: agentic_rag`)

```
[User Question]
      │
      ▼
 [Router Node] ──── needs retrieval? ──No──► [Direct Generate]
      │Yes
      ▼
[Retrieve Node] ──► [Grade Node] ──── pass? ──No──► [Fallback Node]
                                          │Yes
                                          ▼
                                    [Generate Node]
                                          │
                                    [Answer + Sources]
```

**Best for:** Complex queries, multi-hop reasoning, production agents.

---

## REST API

When `serving: fastapi`:

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Health check + config summary |
| `POST` | `/ingest` | Index all documents in `/data` |
| `POST` | `/query` | `{"question": "...", "session_id": "..."}` |
| `GET`  | `/metrics` | Basic usage metrics |
| `GET`  | `/docs` | Swagger UI |

---

## Docker Profiles

| Profile | Command | Activates |
|---------|---------|-----------|
| Default | `docker compose up` | Sandbox + ChromaDB |
| Qdrant | `docker compose --profile qdrant up` | + Qdrant |
| PGVector | `docker compose --profile pgvector up` | + Postgres/pgvector |

---

## CLI Reference

```bash
genai-sandbox new <name> [OPTIONS]   # scaffold a project
genai-sandbox start                  # docker compose up --build
genai-sandbox stop                   # docker compose down
genai-sandbox status                 # docker compose ps
genai-sandbox ingest                 # POST /ingest
genai-sandbox query "your question"  # POST /query
```

---

## Project Structure

```
my-project/
├── config/
│   └── stack.yaml          ← your stack configuration
├── data/                   ← drop PDFs, CSVs, Markdown here
├── src/                    ← your custom scripts (optional)
├── docker/
│   ├── Dockerfile
│   └── entrypoint.sh
├── serving/
│   ├── api.py              ← FastAPI app
│   ├── ui_chainlit.py      ← Chainlit UI
│   └── ui_streamlit.py     ← Streamlit UI
├── templates/
│   ├── naive_rag/
│   │   └── pipeline.py
│   └── agentic_rag/
│       └── pipeline.py
├── docker-compose.yaml
├── Makefile
└── .env.example
```

---

## Supported Stack Combinations

| Orchestrator | Vector DB | LLM | Serving |
|---|---|---|---|
| LangChain | Chroma ✅ | OpenAI ✅ | FastAPI ✅ |
| LangGraph | Qdrant ✅ | Anthropic ✅ | Chainlit ✅ |
| LlamaIndex | PGVector ✅ | Google ✅ | Streamlit ✅ |
| | Pinecone ✅ | Ollama ✅ | Jupyter ✅ |
| | FAISS ✅ | Groq ✅ | |
| | Weaviate ✅ | | |

---

## License

MIT
