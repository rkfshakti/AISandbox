"""
GenAI Sandbox — Chainlit Chat UI
==================================
Provides a production-quality streaming chat interface powered by Chainlit.
Delegates to whichever pipeline is configured (naive_rag | agentic_rag).

Run via:  chainlit run serving/ui_chainlit.py --host 0.0.0.0 --port 8080
Or set:   serving: chainlit  in stack.yaml
"""

from __future__ import annotations

import os
from typing import Any

import chainlit as cl
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

TEMPLATE = os.getenv("SANDBOX_TEMPLATE", os.getenv("TEMPLATE", "naive_rag"))
APP_NAME = "GenAI Sandbox"


# ── Pipeline loader ───────────────────────────────────────────
def _load_pipeline() -> Any:
    if TEMPLATE == "agentic_rag":
        from templates.agentic_rag.pipeline import get_pipeline
    else:
        from templates.naive_rag.pipeline import get_pipeline
    return get_pipeline()


# ── Chainlit lifecycle ────────────────────────────────────────
@cl.on_chat_start
async def on_chat_start():
    """Initialize session state and display a welcome message."""
    pipeline = _load_pipeline()
    cl.user_session.set("pipeline", pipeline)
    cl.user_session.set("session_id", cl.context.session.id)

    await cl.Message(
        content=(
            f"## Welcome to {APP_NAME} 🤖\n\n"
            f"**Template:** `{TEMPLATE}` &nbsp;|&nbsp; "
            f"**Vector DB:** `{os.getenv('SANDBOX_VECTOR_DB', 'chroma')}` &nbsp;|&nbsp; "
            f"**LLM:** `{os.getenv('CHAT_MODEL', 'gpt-4o-mini')}`\n\n"
            "Drop your documents into the `/data` folder and run `/ingest` first.\n\n"
            "**Commands:**\n"
            "- `/ingest` — index documents in /data\n"
            "- `/clear`  — clear conversation history\n"
            "- `/config` — show current stack config\n\n"
            "Then ask any question about your documents!"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming user messages and slash commands."""
    pipeline = cl.user_session.get("pipeline")
    session_id = cl.user_session.get("session_id", "default")
    text = message.content.strip()

    # ── Slash commands ────────────────────────────────────────
    if text.lower() == "/ingest":
        async with cl.Step(name="Indexing documents…") as step:
            try:
                result = pipeline.ingest()
                step.output = f"Indexed **{result.get('indexed', 0)}** chunks from `{result.get('source_files', '?')}` files."
            except Exception as exc:
                step.output = f"Ingest failed: {exc}"
        return

    if text.lower() == "/clear":
        cl.user_session.set("session_id", cl.context.session.id + "_fresh")
        await cl.Message(content="Conversation history cleared.").send()
        return

    if text.lower() == "/config":
        await cl.Message(content=(
            f"**Current Configuration**\n"
            f"- Template: `{TEMPLATE}`\n"
            f"- Orchestrator: `{os.getenv('SANDBOX_ORCHESTRATOR', 'langchain')}`\n"
            f"- Vector DB: `{os.getenv('SANDBOX_VECTOR_DB', 'chroma')}`\n"
            f"- LLM: `{os.getenv('CHAT_MODEL', 'gpt-4o-mini')}`\n"
            f"- Embedding: `{os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')}`\n"
        )).send()
        return

    # ── Main RAG query ────────────────────────────────────────
    msg = cl.Message(content="")
    await msg.send()

    try:
        async with cl.Step(name="Retrieving & generating…") as step:
            result = pipeline.query(text, session_id=session_id)
            answer = result["answer"]
            sources = result.get("sources", [])

            step.output = (
                f"Retrieved **{len(sources)}** source chunk(s). "
                + (f"Grade passed: {result.get('grade_passed')}" if "grade_passed" in result else "")
            )

        # Stream-like update (Chainlit updates in place)
        msg.content = answer

        if sources:
            source_text = "\n\n---\n**Sources:**\n"
            for i, s in enumerate(sources[:3], 1):
                src = s.get("source", "unknown")
                snippet = s.get("content", "")[:200].replace("\n", " ")
                source_text += f"\n**{i}.** `{src}`\n> {snippet}…\n"
            msg.content += source_text

        await msg.update()

    except Exception as exc:
        logger.exception("Query failed in Chainlit handler")
        msg.content = f"An error occurred: {exc}"
        await msg.update()
