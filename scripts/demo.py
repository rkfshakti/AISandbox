#!/usr/bin/env python3
"""
AISandbox — Auto-typing terminal demo for screen recording.
Simulates a real make chat session with realistic delays.
Run: python3 scripts/demo.py
"""
import sys
import time

CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
DIM     = "\033[2m"
BOLD    = "\033[1m"
MAGENTA = "\033[95m"
NC      = "\033[0m"


def write(text: str, end: str = "\n", flush: bool = True) -> None:
    sys.stdout.write(text + end)
    if flush:
        sys.stdout.flush()


def type_out(text: str, delay: float = 0.055) -> None:
    """Simulate human typing character by character."""
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write("\n")
    sys.stdout.flush()


def pause(seconds: float) -> None:
    time.sleep(seconds)


def prompt() -> None:
    write(f"\n{MAGENTA}{BOLD}  You › {NC}", end="", flush=True)


# ── Banner ────────────────────────────────────────────────────
pause(0.3)
write("")
write(f"{CYAN}{BOLD}╔══════════════════════════════════════════════════════╗")
write(f"{CYAN}{BOLD}║          GenAI Sandbox — Terminal Chat               ║")
write(f"{CYAN}{BOLD}╚══════════════════════════════════════════════════════╝{NC}")
write("")
write(f"  {DIM}Provider:{NC}  {BOLD}lm-studio{NC}   {DIM}Model:{NC} {BOLD}qwen3.5-4b{NC}")
write(f"  {DIM}Template:{NC} {BOLD}naive_rag{NC}   {DIM}Vector DB:{NC} {BOLD}chroma{NC}")
write("")
write(f"{DIM}  Type your question, or /help for commands. /quit to exit.{NC}")
write("")
pause(0.8)

# ── Step 1: ingest ────────────────────────────────────────────
prompt()
pause(0.4)
type_out("/ingest", delay=0.10)
pause(0.3)
write(f"\n{YELLOW}  Indexing documents in ./data …{NC}")
pause(1.4)
write(f"{GREEN}  ✓ Indexed 4 chunks from 2 files in 1.1s{NC}")
write("")
pause(0.6)

# ── Step 2: first question ────────────────────────────────────
prompt()
pause(0.5)
type_out("What is RAG and how does it work?", delay=0.065)
pause(0.5)

write("")
write(f"{GREEN}{BOLD}  Assistant")
write(f"{DIM}  ──────────────────────────────────────────────────────{NC}")
pause(1.2)

answer = (
    "Based on the provided context:\n\n"
    "  **What is RAG?**\n"
    "  Retrieval-Augmented Generation (RAG) is a technique that enhances\n"
    "  Large Language Models by combining them with external knowledge\n"
    "  retrieval. Instead of relying solely on training data, RAG systems\n"
    "  dynamically fetch relevant documents and inject them into the LLM's\n"
    "  context window at inference time.\n\n"
    "  **How does it work?**\n"
    "  1. Ingestion  — Documents are chunked, embedded, stored in a vector DB.\n"
    "  2. Retrieval  — Query is embedded; similar chunks fetched via MMR.\n"
    "  3. Generation — Retrieved chunks are injected into the prompt as context."
)
for line in answer.splitlines():
    write("  " + line)
    pause(0.06)

write("")
write(f"{DIM}  Sources (3):")
write(f"{DIM}    • intro_to_rag.md")
write(f"{DIM}    • intro_to_rag.md")
write(f"{DIM}    • README.md{NC}")
write(f"\n{DIM}  ⏱ 16.0s{NC}")
pause(0.8)

# ── Step 3: second question ───────────────────────────────────
prompt()
pause(0.5)
type_out("What are the key benefits of AISandbox?", delay=0.065)
pause(0.5)

write("")
write(f"{GREEN}{BOLD}  Assistant")
write(f"{DIM}  ──────────────────────────────────────────────────────{NC}")
pause(1.0)

answer2 = (
    "Based on the provided context:\n\n"
    "  AISandbox offers three key benefits:\n\n"
    "  • Zero package conflicts — uv manages dependencies dynamically.\n"
    "  • Reproducible environments — same stack.yaml = same behaviour\n"
    "    on any machine.\n"
    "  • Hot reload — edit files in /src and the server reloads\n"
    "    automatically, no container rebuild needed."
)
for line in answer2.splitlines():
    write("  " + line)
    pause(0.06)

write("")
write(f"{DIM}  Sources (2):")
write(f"{DIM}    • intro_to_rag.md")
write(f"{DIM}    • README.md{NC}")
write(f"\n{DIM}  ⏱ 12.3s{NC}")
pause(0.8)

# ── Step 4: /cost ─────────────────────────────────────────────
prompt()
pause(0.4)
type_out("/cost", delay=0.10)
pause(0.3)

write("")
write(f"{CYAN}  Session usage:")
write(f"  Turns:          2")
write(f"  Input tokens:   1,440")
write(f"  Output tokens:  386")
write(f"  Estimated cost: {GREEN}$0.000000  (local model — free){NC}")
write("")
pause(0.7)

# ── Step 5: quit ──────────────────────────────────────────────
prompt()
pause(0.4)
type_out("/quit", delay=0.10)
pause(0.2)
write(f"\n{DIM}  Bye!\n{NC}")
pause(0.5)
