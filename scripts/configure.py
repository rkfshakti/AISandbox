#!/usr/bin/env python3
"""
GenAI Sandbox — Interactive Setup & Configuration Wizard
=========================================================
Run this once to configure your sandbox:

    python3 scripts/configure.py
    — OR —
    make configure

The wizard will:
  1. Ask which LLM provider you want to use
  2. Ask for your API key (hidden input — never echoed)
  3. Ask which orchestrator, vector DB, and serving layer
  4. Write everything to .env and config/stack.yaml
  5. Confirm it's ready to run
"""

from __future__ import annotations

import getpass
import os
import sys
from pathlib import Path

# ── Colour helpers (no external deps needed) ──────────────────
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
NC     = "\033[0m"

def c(color: str, text: str) -> str:
    return f"{color}{text}{NC}"

def banner():
    print()
    print(c(CYAN, BOLD + "╔══════════════════════════════════════════════════╗"))
    print(c(CYAN, BOLD + "║     GenAI Sandbox Wrapper — Setup Wizard         ║"))
    print(c(CYAN, BOLD + "╚══════════════════════════════════════════════════╝"))
    print()

def section(title: str):
    print()
    print(c(YELLOW, f"── {title} " + "─" * (48 - len(title))))

def ask(prompt: str, options: list[tuple[str, str]], default_idx: int = 0) -> tuple[str, str]:
    """Show a numbered menu and return (key, label)."""
    print()
    for i, (key, label) in enumerate(options, 1):
        marker = c(GREEN, " ►") if i == default_idx + 1 else "  "
        print(f"{marker} {c(BOLD, str(i))}) {label}")
    print()
    while True:
        raw = input(f"  {prompt} [default {default_idx + 1}]: ").strip()
        if raw == "":
            return options[default_idx]
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print(c(RED, f"  Invalid — enter a number between 1 and {len(options)}."))

def ask_key(provider_label: str, env_var: str, docs_url: str) -> str:
    """Securely ask for an API key (hidden input)."""
    print()
    print(c(DIM, f"  Get your key at: {docs_url}"))
    while True:
        key = getpass.getpass(f"  Enter your {provider_label} API key: ").strip()
        if key:
            return key
        print(c(RED, "  Key cannot be empty. Try again."))

def confirm(prompt: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    raw = input(f"  {prompt} {suffix}: ").strip().lower()
    if raw == "":
        return default
    return raw in ("y", "yes")

# ══════════════════════════════════════════════════════════════
#  Provider registry
# ══════════════════════════════════════════════════════════════
PROVIDERS = [
    ("openai",        "OpenAI          (gpt-5.4, gpt-5.4-mini, gpt-5.4-nano)"),
    ("anthropic",     "Anthropic       (claude-opus-4-6, claude-sonnet-4-6)"),
    ("google",        "Google          (gemini-2.5-pro, gemini-2.5-flash)"),
    ("groq",          "Groq            (Llama 3.3 70B — ultra-fast)"),
    ("mistral",       "Mistral AI      (Mistral Large, Mistral Small)"),
    ("ollama",        "Ollama          (Local LLMs — Llama 3, Qwen 2.5)"),
    ("cohere",        "Cohere          (Command R+)"),
    ("together",      "Together AI     (100+ open-source models)"),
    ("bedrock",       "AWS Bedrock     (Claude, Titan, Llama via AWS)"),
    ("azure-openai",  "Azure OpenAI    (GPT-4o via Azure)"),
]

PROVIDER_CONFIG = {
    "openai": {
        "env_var":     "OPENAI_API_KEY",
        "docs":        "https://platform.openai.com/api-keys",
        "chat_model":  "gpt-5.4-mini",
        "embed_model": "text-embedding-3-small",
        "extra_vars":  {},
    },
    "anthropic": {
        "env_var":     "ANTHROPIC_API_KEY",
        "docs":        "https://console.anthropic.com/settings/keys",
        "chat_model":  "claude-opus-4-6",
        "embed_model": "text-embedding-3-small",   # use openai for embeddings
        "extra_vars":  {"EMBED_PROVIDER": "openai"},
    },
    "google": {
        "env_var":     "GOOGLE_API_KEY",
        "docs":        "https://aistudio.google.com/app/apikey",
        "chat_model":  "gemini-2.5-pro",
        "embed_model": "models/text-embedding-004",
        "extra_vars":  {},
    },
    "groq": {
        "env_var":     "GROQ_API_KEY",
        "docs":        "https://console.groq.com/keys",
        "chat_model":  "llama-3.3-70b-versatile",
        "embed_model": "text-embedding-3-small",   # use openai for embeddings (groq has no embedding API)
        "extra_vars":  {"EMBED_PROVIDER": "openai"},
    },
    "mistral": {
        "env_var":     "MISTRAL_API_KEY",
        "docs":        "https://console.mistral.ai/api-keys/",
        "chat_model":  "mistral-large-latest",
        "embed_model": "mistral-embed",
        "extra_vars":  {},
    },
    "ollama": {
        "env_var":     "OLLAMA_BASE_URL",
        "docs":        "https://ollama.ai (run  ollama pull llama3  locally)",
        "chat_model":  "llama3.2",
        "embed_model": "nomic-embed-text",
        "extra_vars":  {},
    },
    "cohere": {
        "env_var":     "COHERE_API_KEY",
        "docs":        "https://dashboard.cohere.com/api-keys",
        "chat_model":  "command-r-plus",
        "embed_model": "embed-english-v3.0",
        "extra_vars":  {},
    },
    "together": {
        "env_var":     "TOGETHER_API_KEY",
        "docs":        "https://api.together.xyz/settings/api-keys",
        "chat_model":  "meta-llama/Llama-3-70b-chat-hf",
        "embed_model": "text-embedding-3-small",
        "extra_vars":  {"EMBED_PROVIDER": "openai"},
    },
    "bedrock": {
        "env_var":     "AWS_ACCESS_KEY_ID",
        "docs":        "https://console.aws.amazon.com/iam/home#/security_credentials",
        "chat_model":  "anthropic.claude-opus-4-6-v1",
        "embed_model": "amazon.titan-embed-text-v2:0",
        "extra_vars":  {"AWS_SECRET_ACCESS_KEY": "", "AWS_DEFAULT_REGION": "us-east-1"},
    },
    "azure-openai": {
        "env_var":     "AZURE_OPENAI_API_KEY",
        "docs":        "https://portal.azure.com → Azure OpenAI → Keys",
        "chat_model":  "gpt-5.4",
        "embed_model": "text-embedding-3-small",
        "extra_vars":  {"AZURE_OPENAI_ENDPOINT": "", "AZURE_OPENAI_API_VERSION": "2025-01-01-preview"},
    },
}

ORCHESTRATORS = [
    ("langgraph",   "LangGraph 1.1   (Agentic loops — RECOMMENDED)"),
    ("langchain",   "LangChain 1.2   (Chains & LCEL)"),
    ("llamaindex",  "LlamaIndex 0.14 (Advanced indexing & retrieval)"),
    ("crewai",      "CrewAI 1.14     (Multi-agent crews)"),
    ("autogen",     "AutoGen 0.7     (Conversational agents)"),
    ("dspy",        "DSPy 3.1        (Declarative LM programs)"),
]

VECTOR_DBS = [
    ("chroma",    "ChromaDB  — local, zero-config (RECOMMENDED)"),
    ("qdrant",    "Qdrant    — local or cloud, high performance"),
    ("faiss",     "FAISS     — local, in-memory, no side-car"),
    ("pgvector",  "PGVector  — Postgres (docker compose --profile pgvector)"),
    ("pinecone",  "Pinecone  — cloud vector DB (requires API key)"),
    ("weaviate",  "Weaviate  — cloud vector DB (requires API key)"),
    ("milvus",    "Milvus    — enterprise-grade (docker compose --profile milvus)"),
]

TEMPLATES = [
    ("naive_rag",         "Naive RAG          — Load → Split → Embed → Retrieve → Generate"),
    ("agentic_rag",       "Agentic RAG        — LangGraph Router → Grade → Fallback loop"),
    ("structured_output", "Structured Output  — PydanticAI typed schemas (QA/Summary/Entities)"),
    ("multi_agent",       "Multi-Agent        — Supervisor → Researcher → Critic → Writer"),
    ("mcp_server",        "MCP Server         — Expose knowledge base as MCP tools (Copilot/Claude)"),
]

SERVING_LAYERS = [
    ("fastapi",    "FastAPI    — REST API on :8000  (Swagger at /docs)"),
    ("chainlit",   "Chainlit   — Chat UI on :8080  (conversation history)"),
    ("streamlit",  "Streamlit  — Chat UI on :8501"),
    ("jupyter",    "Jupyter Lab — Notebooks on :8888"),
]

# ══════════════════════════════════════════════════════════════
#  Main wizard
# ══════════════════════════════════════════════════════════════
def main():
    # Locate project root (script lives in scripts/)
    ROOT = Path(__file__).resolve().parent.parent
    ENV_FILE   = ROOT / ".env"
    STACK_FILE = ROOT / "config" / "stack.yaml"

    banner()
    print(c(DIM, "  This wizard configures your GenAI Sandbox in 2 minutes."))
    print(c(DIM, "  Press Enter to accept the default (► marked option)."))

    # ── Step 1: LLM Provider ──────────────────────────────────
    section("Step 1 of 4  —  Choose your LLM Provider")
    provider_key, _ = ask("Choose provider", PROVIDERS, default_idx=0)
    cfg = PROVIDER_CONFIG[provider_key]

    # ── Step 2: API Key ───────────────────────────────────────
    section("Step 2 of 4  —  Enter your API Key")

    if provider_key == "ollama":
        print()
        print(c(DIM, "  Ollama runs locally — no API key needed."))
        print(c(DIM, "  Make sure ollama is running:  ollama serve"))
        api_key = "http://localhost:11434"
        extra_keys: dict[str, str] = {}
    elif provider_key == "bedrock":
        print()
        print(c(DIM, f"  Docs: {cfg['docs']}"))
        api_key = getpass.getpass("  AWS Access Key ID    : ").strip()
        secret   = getpass.getpass("  AWS Secret Access Key: ").strip()
        region   = input("  AWS Region [us-east-1]: ").strip() or "us-east-1"
        extra_keys = {"AWS_SECRET_ACCESS_KEY": secret, "AWS_DEFAULT_REGION": region}
    elif provider_key == "azure-openai":
        print()
        print(c(DIM, f"  Docs: {cfg['docs']}"))
        api_key  = getpass.getpass("  Azure OpenAI API Key  : ").strip()
        endpoint = input("  Azure OpenAI Endpoint : ").strip()
        version  = input("  API Version [2024-08-01-preview]: ").strip() or "2024-08-01-preview"
        extra_keys = {"AZURE_OPENAI_ENDPOINT": endpoint, "AZURE_OPENAI_API_VERSION": version}
    else:
        api_key    = ask_key(provider_key.title(), cfg["env_var"], cfg["docs"])
        extra_keys = {}

    # ── Step 3: Stack choices ─────────────────────────────────
    section("Step 3 of 4  —  Choose your Stack")

    print(c(BOLD, "\n  Orchestrator:"))
    orch_key, _ = ask("Choose orchestrator", ORCHESTRATORS, default_idx=0)

    print(c(BOLD, "\n  Vector Database:"))
    vdb_key, _ = ask("Choose vector DB", VECTOR_DBS, default_idx=0)

    print(c(BOLD, "\n  Template:"))
    tmpl_key, _ = ask("Choose starting template", TEMPLATES, default_idx=0)

    print(c(BOLD, "\n  Serving Layer:"))
    serve_key, _ = ask("Choose serving layer", SERVING_LAYERS, default_idx=0)

    # ── Step 4: Model customisation ───────────────────────────
    section("Step 4 of 4  —  Model Settings")
    print()
    default_chat  = cfg["chat_model"]
    default_embed = cfg["embed_model"]
    print(c(DIM, f"  Default chat model  : {default_chat}"))
    print(c(DIM, f"  Default embed model : {default_embed}"))
    custom = confirm("  Use these defaults?", default=True)
    if not custom:
        chat_model  = input(f"  Chat model  [{default_chat}]: ").strip() or default_chat
        embed_model = input(f"  Embed model [{default_embed}]: ").strip() or default_embed
    else:
        chat_model  = default_chat
        embed_model = default_embed

    # Pinecone / Weaviate cloud keys
    extra_vdb_keys: dict[str, str] = {}
    if vdb_key == "pinecone":
        print()
        pinecone_key = getpass.getpass("  Pinecone API Key: ").strip()
        extra_vdb_keys["PINECONE_API_KEY"] = pinecone_key
    elif vdb_key == "weaviate":
        print()
        weaviate_key = getpass.getpass("  Weaviate API Key: ").strip()
        weaviate_url = input("  Weaviate Cluster URL: ").strip()
        extra_vdb_keys["WEAVIATE_API_KEY"] = weaviate_key
        extra_vdb_keys["WEAVIATE_URL"]     = weaviate_url

    # ── Write .env ────────────────────────────────────────────
    _write_env(
        env_file=ENV_FILE,
        provider=provider_key,
        env_var=cfg["env_var"],
        api_key=api_key,
        extra_provider_keys={**extra_keys, **cfg.get("extra_vars", {})},
        extra_vdb_keys=extra_vdb_keys,
    )

    # ── Write stack.yaml ──────────────────────────────────────
    _write_stack(
        stack_file=STACK_FILE,
        orchestrator=orch_key,
        vector_db=vdb_key,
        llm_provider=provider_key,
        template=tmpl_key,
        serving=serve_key,
        chat_model=chat_model,
        embed_model=embed_model,
    )

    # ── Summary ───────────────────────────────────────────────
    print()
    print(c(GREEN, BOLD + "╔══════════════════════════════════════════════════╗"))
    print(c(GREEN, BOLD + "║  ✓  Configuration saved!                         ║"))
    print(c(GREEN, BOLD + "╚══════════════════════════════════════════════════╝"))
    print()
    print(f"  Provider    : {c(BOLD, provider_key)}")
    print(f"  Orchestrator: {c(BOLD, orch_key)}")
    print(f"  Vector DB   : {c(BOLD, vdb_key)}")
    print(f"  Template    : {c(BOLD, tmpl_key)}")
    print(f"  Serving     : {c(BOLD, serve_key)}")
    print(f"  Chat model  : {c(BOLD, chat_model)}")
    print()
    print(c(CYAN, "  Next steps:"))

    if vdb_key in ("qdrant", "pgvector", "milvus"):
        print(f"  1. make up VECTOR_DB={vdb_key}")
    else:
        profile_hint = "  1. make up"
        if serve_key in ("chainlit", "streamlit"):
            profile_hint += f"         # then open http://localhost:{ '8080' if serve_key == 'chainlit' else '8501' }"
        print(profile_hint)

    print("  2. make ingest         # index your /data documents")
    print("  3. make query Q='your question'")
    print()


# ══════════════════════════════════════════════════════════════
#  File writers
# ══════════════════════════════════════════════════════════════

def _write_env(
    env_file: Path,
    provider: str,
    env_var: str,
    api_key: str,
    extra_provider_keys: dict,
    extra_vdb_keys: dict,
):
    """Rewrite .env — preserves non-provider lines, updates the active key."""
    # Read existing .env if it exists, otherwise use template
    existing: dict[str, str] = {}
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                existing[k.strip()] = v.strip()

    # Apply new values
    existing[env_var] = api_key
    for k, v in extra_provider_keys.items():
        if v:  # only set non-empty extras
            existing[k] = v
    for k, v in extra_vdb_keys.items():
        if v:
            existing[k] = v

    # Always set the provider so entrypoint knows which key to use
    existing["LLM_PROVIDER"] = provider

    lines = [
        "# =============================================================",
        "#  GenAI Sandbox Wrapper — .env  (auto-generated by configure)",
        "#  Re-run:  python3 scripts/configure.py  to change settings",
        "# =============================================================",
        "",
        f"# Active provider: {provider}",
        f"{env_var}={api_key}",
        "",
    ]

    if extra_provider_keys:
        for k, v in extra_provider_keys.items():
            lines.append(f"{k}={v}")
        lines.append("")

    if extra_vdb_keys:
        for k, v in extra_vdb_keys.items():
            lines.append(f"{k}={v}")
        lines.append("")

    lines += [
        "# ── Remaining providers (add keys when needed) ───────────────",
        "OPENAI_API_KEY=" + (existing.get("OPENAI_API_KEY", "") if provider != "openai" else api_key),
        "ANTHROPIC_API_KEY=" + (existing.get("ANTHROPIC_API_KEY", "") if provider != "anthropic" else api_key),
        "GOOGLE_API_KEY=" + (existing.get("GOOGLE_API_KEY", "") if provider != "google" else api_key),
        "GROQ_API_KEY=" + (existing.get("GROQ_API_KEY", "") if provider != "groq" else api_key),
        "MISTRAL_API_KEY=" + (existing.get("MISTRAL_API_KEY", "") if provider != "mistral" else api_key),
        "COHERE_API_KEY=" + (existing.get("COHERE_API_KEY", "") if provider != "cohere" else api_key),
        "TOGETHER_API_KEY=" + (existing.get("TOGETHER_API_KEY", "") if provider != "together" else api_key),
        "PINECONE_API_KEY=" + existing.get("PINECONE_API_KEY", ""),
        "WEAVIATE_API_KEY=" + existing.get("WEAVIATE_API_KEY", ""),
        "",
        "# ── LangSmith Tracing (optional) ─────────────────────────────",
        "LANGCHAIN_TRACING_V2=false",
        "LANGCHAIN_API_KEY=" + existing.get("LANGCHAIN_API_KEY", ""),
        "LANGCHAIN_PROJECT=genai-sandbox",
        "",
        "# ── Port Configuration ────────────────────────────────────────",
        "API_PORT=8000",
        "UI_PORT=8501",
        "CHAINLIT_PORT=8080",
        "JUPYTER_PORT=8888",
        "",
        "# ── Postgres / PGVector ───────────────────────────────────────",
        "POSTGRES_USER=sandbox",
        "POSTGRES_PASSWORD=sandbox",
        "POSTGRES_DB=sandbox_db",
        f"LLM_PROVIDER={provider}",
    ]

    env_file.write_text("\n".join(lines) + "\n")
    print(c(GREEN, f"\n  ✓ Written: {env_file}"))


def _write_stack(
    stack_file: Path,
    orchestrator: str,
    vector_db: str,
    llm_provider: str,
    template: str,
    serving: str,
    chat_model: str,
    embed_model: str,
):
    content = f"""\
# =============================================================
#  GenAI Sandbox — stack.yaml  (auto-generated by configure)
#  Re-run:  python3 scripts/configure.py  to change settings
# =============================================================

orchestrator: {orchestrator}
vector_db: {vector_db}
llm_provider: {llm_provider}
template: {template}
serving: {serving}

embedding_model: {embed_model}
chat_model: {chat_model}
temperature: 0.0
max_tokens: 2048

chunk_size: 1000
chunk_overlap: 150
top_k: 5
collection_name: sandbox_docs
"""
    stack_file.parent.mkdir(parents=True, exist_ok=True)
    stack_file.write_text(content)
    print(c(GREEN, f"  ✓ Written: {stack_file}"))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(c(YELLOW, "\n\n  Setup cancelled."))
        sys.exit(0)
