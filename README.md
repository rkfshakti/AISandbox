# AISandbox

> **`create-react-app` for RAG and Agentic AI. Local-first. Zero boilerplate.**

AISandbox is a scaffold for building RAG and multi-agent AI pipelines.  
Drop documents in a folder, run `make configure`, and in minutes you have a fully working AI pipeline — with your choice of LLM provider (including **100% local via LM Studio / Ollama**), vector database, and template.

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
   4) Groq            (Llama 3.3 70B — ultra-fast inference)
   5) Mistral AI      (Mistral Large, Mistral Small)
   6) Ollama          (Local — Llama 3, Qwen 2.5, Phi-4)
   7) LM Studio       (Local — any GGUF model, OpenAI-compatible)
   8) Cohere          (Command R+)
   9) Together AI     (100+ open-source models)
  10) AWS Bedrock     (Claude, Titan, Llama via AWS)
  11) Azure OpenAI    (gpt-5.4 via Azure)

── Step 2 of 4  —  Enter your API Key ──────────────────

  (For LM Studio / Ollama: press Enter to skip — no key needed)
  Enter your OpenAI API key: **********************

── Step 3 of 4  —  Choose your Stack ───────────────────
  ...

✓ Configuration saved to .env and config/stack.yaml!
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
| **Ollama** | `ollama 0.6.1` | llama3.2, qwen2.5, phi-4 — local |
| **LM Studio** | `openai 2.31` | Any GGUF model — local, OpenAI-compatible API |
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
llm_provider: openai           # openai | anthropic | google | groq | mistral | ollama | lm-studio | cohere | together | bedrock | azure-openai
template: naive_rag            # naive_rag | agentic_rag | structured_output | multi_agent | mcp_server
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

**LM Studio example** (100% local, no API key):
```yaml
llm_provider: lm-studio
chat_model: qwen3.5-4b
embedding_model: text-embedding-nomic-embed-text-v1.5
```
Set `LM_STUDIO_URL=http://localhost:1234/v1` in `.env`.

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

### Template C — Structured Output (`structured_output`)

Uses `.with_structured_output()` to return typed Pydantic models instead of free text.

```bash
# Q&A with confidence score and follow-up questions
curl -X POST /query -d '{"question": "...", "output_schema": "qa"}'

# Bullet-point summary with title
curl -X POST /query -d '{"question": "...", "output_schema": "summary"}'

# Named entity extraction
curl -X POST /query -d '{"question": "...", "output_schema": "entities"}'
```

Best for: Downstream pipelines that need structured data, not prose.

### Template D — Multi-Agent (`multi_agent`)

LangGraph Supervisor pattern:

```
[Question]
     │
[Supervisor] ── retrieval needed? ──No──► [Direct Answer]
     │Yes
[Researcher] → retrieves + drafts
     │
[Critic] ── approved? ──No (up to 2x)──► [Researcher]
     │Yes
[Writer] → polishes final answer
```

Best for: Research-style questions needing fact-checking and polish.

### Template E — MCP Server (`mcp_server`)

Exposes your knowledge base as [Model Context Protocol](https://modelcontextprotocol.io) tools, connectable to VS Code Copilot, Claude Desktop, and Cursor.

```bash
make serve-mcp   # starts stdio MCP server
```

Tools exposed: `search_knowledge_base`, `ingest_documents`, `get_sandbox_status`.

Best for: Giving AI coding assistants access to your private docs.

---

## REST API

When `serving: fastapi` — Swagger UI at [http://localhost:8000/docs](http://localhost:8000/docs)

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check + active config |
| `POST` | `/ingest` | Index all documents in `./data` |
| `POST` | `/query` | `{"question": "...", "session_id": "...", "output_schema": "qa|summary|entities"}` |
| `POST` | `/stream` | Same as `/query` but streams tokens via SSE |
| `GET` | `/metrics` | Requests, latency, token counts, cost estimate |
| `GET` | `/docs` | Swagger UI |

---

## Makefile Commands

```bash
# Setup
make configure          # interactive wizard — choose provider & enter API key
make serve              # run API server locally (no Docker)
make serve-mcp          # run MCP server for VS Code / Claude Desktop / Cursor

# Docker
make up                 # docker compose up --build
make down               # docker compose down

# Data & Queries
make ingest             # index all documents in ./data
make query Q='...'      # ask a question via REST
make stream Q='...'     # stream a response token-by-token (SSE)
make chat               # interactive terminal REPL — no server needed
make eval               # evaluate RAG quality with Ragas metrics

# Observability
make metrics            # token counts, latency, cost estimate
make logs               # stream container logs
make docs               # open Swagger UI in browser
make clean              # remove containers, volumes, cache
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
│   ├── naive_rag/          ← Simple retrieval + generation
│   ├── agentic_rag/        ← LangGraph: router → grade → fallback
│   ├── structured_output/  ← Typed Pydantic responses (qa/summary/entities)
│   ├── multi_agent/        ← Supervisor → Researcher → Critic → Writer
│   └── mcp_server/         ← MCP tools for Copilot/Claude/Cursor
├── src/
│   └── llm_factory.py      ← Multi-provider LLM abstraction (11 providers)
├── scripts/
│   ├── configure.py        ← Interactive setup wizard
│   ├── chat.py             ← Terminal REPL (no server needed)
│   └── eval.py             ← Ragas evaluation (faithfulness, relevance, precision)
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

*Built with Python 3.13+ · Powered by LangChain, LangGraph, and the full 2026 AI ecosystem.*
