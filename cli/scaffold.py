#!/usr/bin/env python3
"""
genai-sandbox CLI
==================
A create-react-app-style scaffolding tool for the GenAI Sandbox Wrapper.

Usage:
    genai-sandbox new my-project
    genai-sandbox new my-project --template agentic_rag --vector-db qdrant --serving chainlit
    genai-sandbox start
    genai-sandbox ingest
    genai-sandbox status
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# ── Where the wrapper's template assets live ──────────────────
PACKAGE_DIR = Path(__file__).parent.parent  # genai-sandbox root


# ══════════════════════════════════════════════════════════════
#  CLI group
# ══════════════════════════════════════════════════════════════

@click.group()
@click.version_option(version="1.0.0", prog_name="genai-sandbox")
def cli():
    """GenAI Sandbox Wrapper — scaffold and run RAG / Agentic AI environments."""


# ══════════════════════════════════════════════════════════════
#  genai-sandbox new <project-name>
# ══════════════════════════════════════════════════════════════

@cli.command()
@click.argument("project_name")
@click.option("--template",   default="naive_rag",
              type=click.Choice(["naive_rag", "agentic_rag"]),
              show_default=True, help="Starting template.")
@click.option("--orchestrator", default="langchain",
              type=click.Choice(["langchain", "langgraph", "llamaindex"]),
              show_default=True)
@click.option("--vector-db",  default="chroma",
              type=click.Choice(["chroma", "qdrant", "pgvector", "pinecone", "faiss"]),
              show_default=True)
@click.option("--llm",        default="openai",
              type=click.Choice(["openai", "anthropic", "google", "ollama", "groq"]),
              show_default=True, help="LLM provider.")
@click.option("--serving",    default="fastapi",
              type=click.Choice(["fastapi", "streamlit", "chainlit", "jupyter"]),
              show_default=True)
def new(project_name, template, orchestrator, vector_db, llm, serving):
    """Scaffold a new GenAI sandbox project."""
    target = Path.cwd() / project_name

    # ── Guard against overwriting ──────────────────────────────
    if target.exists():
        console.print(f"[red]Error:[/red] Directory '{project_name}' already exists.")
        sys.exit(1)

    console.print(Panel.fit(
        f"[bold cyan]Creating GenAI Sandbox Project[/bold cyan]\n\n"
        f"  Name        : [green]{project_name}[/green]\n"
        f"  Template    : [yellow]{template}[/yellow]\n"
        f"  Orchestrator: [yellow]{orchestrator}[/yellow]\n"
        f"  Vector DB   : [yellow]{vector_db}[/yellow]\n"
        f"  LLM         : [yellow]{llm}[/yellow]\n"
        f"  Serving     : [yellow]{serving}[/yellow]",
        title="genai-sandbox"
    ))

    # ── Copy project skeleton ──────────────────────────────────
    skeleton_src = PACKAGE_DIR
    dirs_to_copy = ["docker", "serving", "templates", "scripts"]

    target.mkdir(parents=True)
    for d in dirs_to_copy:
        src = skeleton_src / d
        if src.exists():
            shutil.copytree(src, target / d)

    # ── Create user-facing volume dirs ────────────────────────
    for d in ["data", "src", "config"]:
        (target / d).mkdir(exist_ok=True)
        (target / d / ".gitkeep").touch()

    # ── Write stack.yaml from options ─────────────────────────
    stack = {
        "orchestrator": orchestrator,
        "vector_db": vector_db,
        "llm_provider": llm,
        "template": template,
        "serving": serving,
        "embedding_model": "text-embedding-3-small",
        "chat_model": "gpt-4o-mini",
        "temperature": 0.0,
        "max_tokens": 2048,
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "top_k": 5,
        "collection_name": f"{project_name.replace('-', '_')}_docs",
    }
    with open(target / "config" / "stack.yaml", "w") as f:
        yaml.dump(stack, f, sort_keys=False, default_flow_style=False)

    # ── Copy .env.example ─────────────────────────────────────
    env_src = PACKAGE_DIR / ".env.example"
    if env_src.exists():
        shutil.copy(env_src, target / ".env.example")

    # ── Write docker-compose override pointing to new config ──
    compose_src = skeleton_src / "docker" / "docker-compose.yaml"
    if compose_src.exists():
        shutil.copy(compose_src, target / "docker-compose.yaml")

    # ── Write Makefile for convenience ────────────────────────
    makefile = target / "Makefile"
    makefile.write_text(_makefile_content(project_name, vector_db))

    # ── Write .gitignore ──────────────────────────────────────
    (target / ".gitignore").write_text(
        ".env\n__pycache__/\n*.pyc\n.chroma/\n.faiss/\ndata/*\n!data/.gitkeep\n"
    )

    # ── Done ──────────────────────────────────────────────────
    console.print(f"\n[bold green]✓ Project created:[/bold green] {target}\n")
    _print_next_steps(project_name, serving, vector_db)


def _makefile_content(name: str, vector_db: str) -> str:
    profiles = f"--profile {vector_db}" if vector_db in ("qdrant", "pgvector") else ""
    return f"""\
.PHONY: up down logs ingest shell

up:
\tdocker compose {profiles} up --build -d

down:
\tdocker compose down

logs:
\tdocker compose logs -f sandbox

ingest:
\tcurl -s -X POST http://localhost:8000/ingest | python3 -m json.tool

query:
\t@read -p "Question: " q; curl -s -X POST http://localhost:8000/query -H 'Content-Type: application/json' -d "{\\"question\\":\\"$$q\\"}" | python3 -m json.tool

shell:
\tdocker exec -it genai-sandbox bash

status:
\tdocker compose ps
"""


def _print_next_steps(name: str, serving: str, vector_db: str):
    table = Table(title="Next Steps", show_header=False, box=None)
    table.add_column(style="bold cyan", width=4)
    table.add_column()

    steps = [
        ("1.", f"cd {name}"),
        ("2.", "cp .env.example .env   # add your API keys"),
        ("3.", "make up                # build & start the sandbox"),
    ]

    if vector_db in ("qdrant", "pgvector"):
        steps.insert(2, ("  ", f"# docker compose --profile {vector_db} up -d  (side-car DB)"))

    steps += [
        ("4.", "make ingest            # index your /data documents"),
        ("5.", "make query             # ask a question"),
    ]

    if serving == "streamlit":
        steps.append(("  ", "open http://localhost:8501"))
    elif serving == "chainlit":
        steps.append(("  ", "open http://localhost:8080"))
    else:
        steps.append(("  ", "open http://localhost:8000/docs"))

    for n, s in steps:
        table.add_row(n, s)

    console.print(table)


# ══════════════════════════════════════════════════════════════
#  genai-sandbox start  (must be run inside a scaffolded project)
# ══════════════════════════════════════════════════════════════

@cli.command()
@click.option("--profile", default=None, help="Docker compose profile (e.g., qdrant, pgvector).")
@click.option("--build/--no-build", default=True, help="Force image rebuild.")
def start(profile, build):
    """Start the sandbox (docker compose up)."""
    _require_compose_file()
    cmd = ["docker", "compose"]
    if profile:
        cmd += ["--profile", profile]
    cmd += ["up", "-d"]
    if build:
        cmd.append("--build")
    console.print(f"[cyan]Running:[/cyan] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    console.print("[green]Sandbox is up.[/green]")


@cli.command()
def stop():
    """Stop the sandbox (docker compose down)."""
    _require_compose_file()
    subprocess.run(["docker", "compose", "down"], check=True)


@cli.command()
def status():
    """Show sandbox container status."""
    _require_compose_file()
    subprocess.run(["docker", "compose", "ps"], check=True)


@cli.command()
@click.option("--port", default=8000, show_default=True)
def ingest(port):
    """Trigger document ingestion via the sandbox API."""
    import httpx
    url = f"http://localhost:{port}/ingest"
    console.print(f"[cyan]POST[/cyan] {url}")
    try:
        r = httpx.post(url, timeout=120)
        r.raise_for_status()
        console.print_json(r.text)
    except Exception as exc:
        console.print(f"[red]Failed:[/red] {exc}")
        sys.exit(1)


@cli.command()
@click.argument("question")
@click.option("--port", default=8000, show_default=True)
@click.option("--session-id", default="cli-session", show_default=True)
def query(question, port, session_id):
    """Send a question to the sandbox API."""
    import httpx
    url = f"http://localhost:{port}/query"
    console.print(f"[cyan]POST[/cyan] {url}")
    try:
        r = httpx.post(url, json={"question": question, "session_id": session_id}, timeout=120)
        r.raise_for_status()
        data = r.json()
        console.print(Panel(data["answer"], title="Answer", style="green"))
        if data.get("sources"):
            console.print(f"\n[dim]Sources: {len(data['sources'])} chunk(s)[/dim]")
    except Exception as exc:
        console.print(f"[red]Failed:[/red] {exc}")
        sys.exit(1)


# ── Helpers ───────────────────────────────────────────────────
def _require_compose_file():
    if not Path("docker-compose.yaml").exists():
        console.print("[red]Error:[/red] No docker-compose.yaml found. Run this from inside a scaffolded project.")
        sys.exit(1)


if __name__ == "__main__":
    cli()
