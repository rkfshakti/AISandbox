# =============================================================
#  GenAI Sandbox Wrapper — Makefile
#  Run all commands from the genai-sandbox/ directory.
# =============================================================

COMPOSE     = docker compose
SANDBOX     = genai-sandbox
API_URL     = http://localhost:8000
VECTOR_DB  ?= chroma

.DEFAULT_GOAL := help

# ── Detect optional profiles ──────────────────────────────────
PROFILES =
ifneq ($(filter qdrant pgvector ollama milvus,$(VECTOR_DB)),)
    PROFILES = --profile $(VECTOR_DB)
endif

# ── Help ──────────────────────────────────────────────────────
.PHONY: help
help:
	@echo ""
	@echo "  GenAI Sandbox Wrapper — Available Commands"
	@echo "  ───────────────────────────────────────────"
	@echo "  ── Setup & Serving ──────────────────────────"
	@echo "  make configure      Interactive setup — choose provider & API key"
	@echo "  make serve          Run the API server locally (no Docker)"
	@echo "  make serve-mcp      Run the MCP server (stdio, for AI clients)"
	@echo "  make chat           Interactive terminal chat (no server needed)"
	@echo "  make up             Build & start all containers"
	@echo "  make down           Stop and remove containers"
	@echo ""
	@echo "  ── Data & Queries ───────────────────────────"
	@echo "  make ingest         Index all documents in ./data"
	@echo "  make query Q='...'  Ask a question (REST)"
	@echo "  make stream Q='...' Stream a response token-by-token (SSE)"
	@echo "  make eval           Evaluate RAG quality with Ragas"
	@echo ""
	@echo "  ── Observability ────────────────────────────"
	@echo "  make metrics        Show API usage, token counts & cost"
	@echo "  make docs           Open Swagger UI in browser"
	@echo "  make logs           Stream sandbox logs"
	@echo "  make status         Show container health"
	@echo ""
	@echo "  ── Templates ────────────────────────────────"
	@echo "  naive_rag           Simple retrieval + generation"
	@echo "  agentic_rag         LangGraph: router → grade → fallback"
	@echo "  structured_output   PydanticAI: typed schema responses"
	@echo "  multi_agent         Supervisor → Researcher → Critic → Writer"
	@echo "  mcp_server          MCP tools for Copilot/Claude/Cursor"
	@echo ""
	@echo "  Switch:  make configure  (or edit TEMPLATE in config/stack.yaml)"
	@echo ""

# ── Interactive configure ─────────────────────────────────────
.PHONY: configure
configure:
	@if [ -f .venv/bin/python3 ]; then \
		.venv/bin/python3 scripts/configure.py; \
	else \
		python3 scripts/configure.py; \
	fi

# ── Run locally (no Docker) ───────────────────────────────────
.PHONY: serve
serve:
	@echo "Starting GenAI Sandbox API on http://localhost:8000 ..."
	@PYTHONPATH=. uvicorn serving.api:app --host 0.0.0.0 --port 8000 --reload

# ── Setup (legacy — use make configure instead) ───────────────
.PHONY: setup
setup:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✓ .env created — run  make configure  to enter your API key."; \
	else \
		echo "✓ .env already exists.  Run  make configure  to change settings."; \
	fi

# ── Build & Start ─────────────────────────────────────────────
.PHONY: up
up: setup
	$(COMPOSE) $(PROFILES) up --build -d
	@echo ""
	@echo "  ✓ Sandbox is running!"
	@echo "  FastAPI  → $(API_URL)      (docs: $(API_URL)/docs)"
	@echo "  Streamlit→ http://localhost:8501"
	@echo "  Chainlit → http://localhost:8080"
	@echo "  Jupyter  → http://localhost:8888"
	@echo ""
	@echo "  Next: make ingest"

.PHONY: down
down:
	$(COMPOSE) down

.PHONY: restart
restart:
	$(COMPOSE) restart sandbox

.PHONY: rebuild
rebuild:
	$(COMPOSE) $(PROFILES) up --build --force-recreate -d

# ── Logs ──────────────────────────────────────────────────────
.PHONY: logs
logs:
	$(COMPOSE) logs -f sandbox

.PHONY: logs-all
logs-all:
	$(COMPOSE) logs -f

# ── Status ────────────────────────────────────────────────────
.PHONY: status
status:
	$(COMPOSE) ps
	@echo ""
	@curl -s $(API_URL)/ | python3 -m json.tool 2>/dev/null || echo "API not yet ready."

# ── Ingest ────────────────────────────────────────────────────
.PHONY: ingest
ingest:
	@echo "Indexing documents in ./data …"
	@curl -s -X POST $(API_URL)/ingest \
		-H "Content-Type: application/json" \
		| python3 -m json.tool

# ── Query ─────────────────────────────────────────────────────
.PHONY: query
query:
ifndef Q
	@echo "Usage: make query Q='your question here'"
else
	@curl -s -X POST $(API_URL)/query \
		-H "Content-Type: application/json" \
		-d '{"question": "$(Q)", "session_id": "makefile"}' \
		| python3 -m json.tool
endif
# ── Stream (SSE) ───────────────────────────────────────
.PHONY: stream
stream:
ifndef Q
	@echo "Usage: make stream Q='your question here'"
else
	@echo "Streaming response (Ctrl+C to stop):"
	@curl -N -s -X POST $(API_URL)/stream \
		-H "Content-Type: application/json" \
		-d '{"question": "$(Q)", "session_id": "makefile"}'
	@echo ""
endif

# ── RAG Evaluation ────────────────────────────────────
.PHONY: eval
eval:
	@echo "Running RAG evaluation (API must be running: make serve)"
	@if [ -f .venv/bin/python3 ]; then \
		.venv/bin/python3 scripts/eval.py; \
	else \
		python3 scripts/eval.py; \
	fi

# ── Terminal Chat (no server needed) ────────────────────
.PHONY: chat
chat:
	@if [ -f .venv/bin/python3 ]; then \
		PYTHONPATH=. .venv/bin/python3 scripts/chat.py; \
	else \
		PYTHONPATH=. python3 scripts/chat.py; \
	fi

# ── MCP Server (stdio for AI clients) ───────────────────
.PHONY: serve-mcp
serve-mcp:
	@echo "Starting MCP server (stdio transport)..."
	@echo "Connect via Claude Desktop or VS Code MCP extension."
	@if [ -f .venv/bin/python3 ]; then \
		PYTHONPATH=. .venv/bin/python3 -m templates.mcp_server.pipeline; \
	else \
		PYTHONPATH=. python3 -m templates.mcp_server.pipeline; \
	fi
# ── Metrics ───────────────────────────────────────────────────
.PHONY: metrics
metrics:
	@curl -s $(API_URL)/metrics | python3 -m json.tool

# ── Shell ─────────────────────────────────────────────────────
.PHONY: shell
shell:
	docker exec -it $(SANDBOX) bash

# ── Open docs ─────────────────────────────────────────────────
.PHONY: docs
docs:
	@open $(API_URL)/docs 2>/dev/null || xdg-open $(API_URL)/docs

# ── Clean ─────────────────────────────────────────────────────
.PHONY: clean
clean:
	$(COMPOSE) down -v --remove-orphans
	docker rmi genai-sandbox:latest 2>/dev/null || true
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "✓ Cleaned."
