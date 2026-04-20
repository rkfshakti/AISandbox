#!/usr/bin/env python3
"""
GenAI Sandbox — Interactive Terminal Chat
==========================================
A zero-server chat loop for exploring your documents locally.
No API server needed — runs the pipeline directly in-process.

Usage:
    python3 scripts/chat.py
    — or —
    make chat

Commands during chat:
    /help       show available commands
    /ingest     re-index documents in ./data
    /clear      reset conversation history
    /status     show current stack config
    /cost       show token usage and estimated cost so far
    /quit       exit

Supports all configured LLM providers (openai, anthropic, ollama, etc.)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# ── Path bootstrap ─────────────────────────────────────────────
HERE = Path(__file__).resolve().parent.parent  # genai-sandbox/
sys.path.insert(0, str(HERE))
load_dotenv(HERE / ".env")

# ── ANSI colours ──────────────────────────────────────────────
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
MAGENTA = "\033[95m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
NC      = "\033[0m"


def c(color: str, text: str) -> str:
    return f"{color}{text}{NC}"


# ── Session cost tracker ──────────────────────────────────────
_session: dict = {
    "turns":         0,
    "input_tokens":  0,
    "output_tokens": 0,
    "cost_usd":      0.0,
}


def _banner() -> None:
    provider = os.getenv("LLM_PROVIDER", "openai")
    model    = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    vdb      = os.getenv("SANDBOX_VECTOR_DB", "chroma")
    template = os.getenv("SANDBOX_TEMPLATE", os.getenv("TEMPLATE", "naive_rag"))

    print()
    print(c(CYAN, BOLD + "╔══════════════════════════════════════════════════════╗"))
    print(c(CYAN, BOLD + "║          GenAI Sandbox — Terminal Chat               ║"))
    print(c(CYAN, BOLD + "╚══════════════════════════════════════════════════════╝"))
    print()
    print(f"  {c(DIM, 'Provider:')}  {c(BOLD, provider)}   {c(DIM, 'Model:')} {c(BOLD, model)}")
    print(f"  {c(DIM, 'Template:')} {c(BOLD, template)}   {c(DIM, 'Vector DB:')} {c(BOLD, vdb)}")
    print()
    print(c(DIM, "  Type your question, or /help for commands. /quit to exit."))
    print()


def _help() -> None:
    print(c(CYAN, "\n  Available commands:"))
    cmds = [
        ("/ingest",  "Re-index all documents in ./data"),
        ("/clear",   "Clear conversation history"),
        ("/status",  "Show current stack configuration"),
        ("/cost",    "Show token usage and estimated cost"),
        ("/quit",    "Exit the chat"),
    ]
    for cmd, desc in cmds:
        print(f"  {c(BOLD, cmd):<20} {desc}")
    print()


def _show_status() -> None:
    print()
    items = [
        ("Provider",   os.getenv("LLM_PROVIDER", "openai")),
        ("Model",      os.getenv("CHAT_MODEL", "gpt-4o-mini")),
        ("Embed",      os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")),
        ("Vector DB",  os.getenv("SANDBOX_VECTOR_DB", "chroma")),
        ("Template",   os.getenv("SANDBOX_TEMPLATE", os.getenv("TEMPLATE", "naive_rag"))),
        ("Data dir",   os.getenv("DATA_DIR", str(HERE / "data"))),
    ]
    for label, val in items:
        print(f"  {c(DIM, label + ':')} {val}")
    print()


def _show_cost() -> None:
    print()
    print(c(CYAN, "  Session usage:"))
    print(f"  Turns:         {_session['turns']}")
    print(f"  Input tokens:  {_session['input_tokens']:,}")
    print(f"  Output tokens: {_session['output_tokens']:,}")
    cost_col = GREEN if _session["cost_usd"] < 0.01 else YELLOW
    print(f"  Estimated cost: {c(cost_col, f'${_session[\"cost_usd\"]:.6f}')}")
    print()


def _load_pipeline():
    """Load the configured pipeline template."""
    template = os.getenv("SANDBOX_TEMPLATE", os.getenv("TEMPLATE", "naive_rag"))
    if template == "agentic_rag":
        from templates.agentic_rag.pipeline import get_pipeline
    elif template == "structured_output":
        from templates.structured_output.pipeline import get_pipeline
    elif template == "multi_agent":
        from templates.multi_agent.pipeline import get_pipeline
    else:
        from templates.naive_rag.pipeline import get_pipeline
    return get_pipeline()


def _do_ingest(pipeline) -> None:
    print(c(YELLOW, "\n  Indexing documents in ./data …"))
    t0 = time.perf_counter()
    try:
        result  = pipeline.ingest()
        elapsed = time.perf_counter() - t0
        indexed = result.get("indexed", 0)
        files   = result.get("source_files", "?")
        print(c(GREEN, f"  ✓ Indexed {indexed} chunks from {files} files in {elapsed:.1f}s"))
    except Exception as exc:
        print(c(RED, f"  ✗ Ingest failed: {exc}"))
    print()


def _do_query(pipeline, question: str, history: list[dict]) -> str:
    """Run a query and return the answer. Updates _session cost tracking."""
    t0 = time.perf_counter()
    try:
        result  = pipeline.query(question, session_id="chat")
        elapsed = time.perf_counter() - t0
        answer  = result.get("answer", "(no answer)")
        sources = result.get("sources", [])

        # Cost tracking
        from src.llm_factory import estimate_cost
        model         = os.getenv("CHAT_MODEL", "gpt-4o-mini")
        input_tokens  = result.get("input_tokens", 0)
        output_tokens = result.get("output_tokens", 0)
        if input_tokens or output_tokens:
            _session["input_tokens"]  += input_tokens
            _session["output_tokens"] += output_tokens
            _session["cost_usd"]      += estimate_cost(model, input_tokens, output_tokens)
        _session["turns"] += 1

        # Print answer
        print()
        print(c(GREEN, BOLD + "  Assistant"))
        print(c(DIM, "  " + "─" * 54))
        # Word-wrap answer at 70 chars for readability
        words, line = answer.split(), ""
        for word in words:
            if len(line) + len(word) + 1 > 72:
                print(f"  {line}")
                line = word
            else:
                line = (line + " " + word).strip()
        if line:
            print(f"  {line}")

        # Sources
        if sources:
            print()
            print(c(DIM, f"  Sources ({len(sources)}):"))
            for s in sources[:3]:
                src = s.get("source", "unknown")
                pg  = f" p.{s['page']}" if s.get("page") else ""
                print(c(DIM, f"    • {Path(src).name}{pg}"))

        print(c(DIM, f"\n  ⏱ {elapsed:.2f}s"))
        return answer

    except Exception as exc:
        print(c(RED, f"\n  ✗ Error: {exc}"))
        return ""


def main() -> None:
    _banner()

    # ── Load pipeline ─────────────────────────────────────────
    print(c(DIM, "  Loading pipeline…"), end="", flush=True)
    try:
        pipeline = _load_pipeline()
        print(c(GREEN, " ready.\n"))
    except Exception as exc:
        print(c(RED, f"\n  ✗ Failed to load pipeline: {exc}"))
        print(c(DIM, "  Run  make configure  to set up your provider."))
        sys.exit(1)

    history: list[dict] = []

    while True:
        try:
            raw = input(c(MAGENTA, BOLD + "\n  You › ") + NC).strip()
        except (KeyboardInterrupt, EOFError):
            print(c(DIM, "\n\n  Bye!\n"))
            _show_cost()
            break

        if not raw:
            continue

        # ── Commands ──────────────────────────────────────────
        cmd = raw.lower()

        if cmd in ("/quit", "/exit", "/q"):
            print(c(DIM, "\n  Bye!\n"))
            _show_cost()
            break

        if cmd == "/help":
            _help()
            continue

        if cmd == "/status":
            _show_status()
            continue

        if cmd == "/cost":
            _show_cost()
            continue

        if cmd == "/clear":
            history.clear()
            print(c(GREEN, "  ✓ History cleared."))
            continue

        if cmd == "/ingest":
            _do_ingest(pipeline)
            continue

        # ── Regular question ──────────────────────────────────
        answer = _do_query(pipeline, raw, history)
        if answer:
            history.append({"role": "user",      "content": raw})
            history.append({"role": "assistant",  "content": answer})


if __name__ == "__main__":
    main()
