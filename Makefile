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
	@echo "  make configure      Interactive setup — choose provider & enter API key"
	@echo "  make serve          Run the API server locally (no Docker needed)"
	@echo "  make up             Build & start all containers"
	@echo "  make down           Stop and remove containers"
	@echo "  make restart        Restart the sandbox container"
	@echo "  make logs           Stream sandbox logs"
	@echo "  make ingest         Index all documents in ./data"
	@echo "  make query Q='...'  Ask a question"
	@echo "  make status         Show container health"
	@echo "  make shell          Open bash inside the sandbox"
	@echo "  make clean          Remove containers, volumes, and cache"
	@echo "  make rebuild        Force full image rebuild"
	@echo "  make metrics        Show API usage metrics"
	@echo "  make docs           Open Swagger UI in browser"
	@echo ""
	@echo "  Vector DB profiles: VECTOR_DB=qdrant|pgvector|ollama|milvus"
	@echo "  Example:  make up VECTOR_DB=qdrant"
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
