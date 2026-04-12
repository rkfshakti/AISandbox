"""
GenAI Sandbox — Streamlit Chat UI
====================================
A clean, conversational chat interface for the sandbox.
Delegates to whichever pipeline is configured (naive_rag | agentic_rag).

Run via:  streamlit run serving/ui_streamlit.py
Or set:   serving: streamlit  in stack.yaml
"""

from __future__ import annotations

import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

TEMPLATE = os.getenv("SANDBOX_TEMPLATE", os.getenv("TEMPLATE", "naive_rag"))
VECTOR_DB = os.getenv("SANDBOX_VECTOR_DB", "chroma")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
ORCHESTRATOR = os.getenv("SANDBOX_ORCHESTRATOR", "langchain")

st.set_page_config(
    page_title="GenAI Sandbox",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Pipeline (cached across reruns) ───────────────────────────
@st.cache_resource(show_spinner="Loading pipeline…")
def _load_pipeline() -> Any:
    if TEMPLATE == "agentic_rag":
        from templates.agentic_rag.pipeline import get_pipeline
    else:
        from templates.naive_rag.pipeline import get_pipeline
    return get_pipeline()


# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/openai.svg", width=40)
    st.title("GenAI Sandbox")

    st.markdown("### Stack Config")
    st.code(
        f"template:     {TEMPLATE}\n"
        f"orchestrator: {ORCHESTRATOR}\n"
        f"vector_db:    {VECTOR_DB}\n"
        f"llm:          {CHAT_MODEL}",
        language="yaml",
    )

    st.markdown("---")
    st.markdown("### Document Ingestion")
    st.info("Drop files into the `/data` folder, then click **Ingest**.")
    if st.button("⚡ Ingest /data Documents", use_container_width=True):
        with st.spinner("Indexing…"):
            pipeline = _load_pipeline()
            result = pipeline.ingest()
            indexed = result.get("indexed", 0)
            files = result.get("source_files", "?")
            if indexed:
                st.success(f"Indexed {indexed} chunks from {files} file(s).")
            else:
                st.warning(result.get("message", "No documents found in /data."))

    st.markdown("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("GenAI Sandbox Wrapper · v1.0")


# ── Chat area ──────────────────────────────────────────────────
st.title("🤖 GenAI Sandbox Chat")
st.caption(f"Powered by **{TEMPLATE}** · **{VECTOR_DB}** · **{CHAT_MODEL}**")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I'm your GenAI sandbox assistant.\n\n"
                "1. Drop your documents into the `/data` folder.\n"
                "2. Click **Ingest /data Documents** in the sidebar.\n"
                "3. Ask me anything about them!"
            ),
        }
    ]

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📄 {len(msg['sources'])} source(s)"):
                for s in msg["sources"]:
                    st.markdown(f"**`{s.get('source', 'unknown')}`**")
                    st.markdown(f"> {s.get('content', '')[:250]}…")

# User input
if prompt := st.chat_input("Ask a question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                pipeline = _load_pipeline()
                result = pipeline.query(prompt, session_id="streamlit-session")
                answer = result["answer"]
                sources = result.get("sources", [])

                st.markdown(answer)

                if sources:
                    with st.expander(f"📄 {len(sources)} source(s) used"):
                        for s in sources:
                            st.markdown(f"**`{s.get('source', 'unknown')}`**")
                            st.markdown(f"> {s.get('content', '')[:250]}…")

                meta_parts = []
                if result.get("retrieval_used") is not None:
                    meta_parts.append(f"retrieval: {'yes' if result['retrieval_used'] else 'no'}")
                if result.get("grade_passed") is not None:
                    meta_parts.append(f"grade: {'pass' if result['grade_passed'] else 'fail'}")
                if meta_parts:
                    st.caption(" · ".join(meta_parts))

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as exc:
                error_msg = f"Error: {exc}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
