"""
Template E — Multi-Agent RAG (LangGraph supervisor pattern)
============================================================
A supervisor agent coordinates a team of specialist agents:

  Supervisor
    ├── Researcher   — retrieves and synthesises document chunks
    ├── Critic       — fact-checks the researcher's answer against context
    └── Writer       — rewrites the final answer for clarity and conciseness

Graph topology:

    [START]
       │
  [supervisor] ──► [researcher] ──► [critic] ──► [writer] ──► [END]
       │                                  │
       └─────── (no retrieval needed) ────┘

The supervisor decides whether retrieval is needed. If not, it routes
directly to writer. If the critic finds hallucinations, it loops back
to researcher (up to MAX_ITERATIONS times).

Usage:
    POST /ingest  → index documents
    POST /query   → {"question": "...", "session_id": "..."}

Run standalone:
    from templates.multi_agent.pipeline import get_pipeline
    p = get_pipeline()
    p.ingest()
    result = p.query("Compare the techniques described in the documents.")
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from loguru import logger

HERE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(HERE))
load_dotenv(HERE / ".env")

from src.llm_factory import build_embeddings, build_llm  # noqa: E402

# ── Settings ──────────────────────────────────────────────────
DATA_DIR       = Path(os.getenv("DATA_DIR", str(HERE / "data")))
COLLECTION     = os.getenv("COLLECTION_NAME", "sandbox_docs")
CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K          = int(os.getenv("TOP_K", "5"))
MAX_ITERATIONS = 2  # critic → researcher loop limit


# ══════════════════════════════════════════════════════════════
#  Graph State
# ══════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    question:           str
    documents:          list[Document]
    researcher_draft:   str
    critic_feedback:    str
    final_answer:       str
    retrieval_needed:   bool
    critic_approved:    bool
    iterations:         int


# ── Vector store ──────────────────────────────────────────────
def _build_vectorstore(embeddings: Any) -> Any:
    vdb = os.getenv("SANDBOX_VECTOR_DB", "chroma").lower()
    if vdb == "chroma":
        from langchain_chroma import Chroma
        return Chroma(
            collection_name=COLLECTION,
            embedding_function=embeddings,
            persist_directory=str(DATA_DIR / ".chroma"),
        )
    if vdb == "qdrant":
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
        return QdrantVectorStore(
            client=QdrantClient(url=os.getenv("QDRANT_URL", "http://genai-qdrant:6333")),
            collection_name=COLLECTION,
            embedding=embeddings,
        )
    raise ValueError(f"Unsupported vector_db: {vdb}")


def _load_documents() -> list[Document]:
    from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
    LOADER_MAP = {
        ".pdf": PyPDFLoader, ".csv": CSVLoader,
        ".md": TextLoader,   ".txt": TextLoader,
        ".py": TextLoader,   ".json": TextLoader,
    }
    docs: list[Document] = []
    for path in DATA_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in LOADER_MAP:
            try:
                docs.extend(LOADER_MAP[path.suffix.lower()](str(path)).load())
            except Exception as exc:
                logger.warning(f"Skipping {path.name}: {exc}")
    return docs


# ══════════════════════════════════════════════════════════════
#  Agent Node Factories (each closes over the shared LLM)
# ══════════════════════════════════════════════════════════════

def _make_supervisor_node(llm: Any):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a supervisor agent. Decide if the question needs document retrieval "
         "from the knowledge base, or can be answered from general knowledge.\n"
         'Reply with JSON: {{"retrieval_needed": true|false, "reason": "..."}}'),
        ("human", "{question}"),
    ])
    chain = prompt | llm | JsonOutputParser()

    def supervisor_node(state: AgentState) -> AgentState:
        result = chain.invoke({"question": state["question"]})
        retrieval_needed = result.get("retrieval_needed", True)
        logger.info(f"[Supervisor] retrieval_needed={retrieval_needed}: {result.get('reason', '')}")
        return {**state, "retrieval_needed": retrieval_needed}

    return supervisor_node


def _make_researcher_node(llm: Any, retriever: Any):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a research agent with access to a knowledge base. "
         "Use the provided context to write a thorough, accurate draft answer. "
         "Include specific details and cite source names where possible.\n\n"
         "Context:\n{context}"),
        ("human", "Question: {question}\n\nCritic feedback (if any): {feedback}"),
    ])
    chain = prompt | llm | StrOutputParser()

    def researcher_node(state: AgentState) -> AgentState:
        docs     = retriever.invoke(state["question"])
        context  = "\n\n---\n\n".join(d.page_content for d in docs)
        feedback = state.get("critic_feedback", "None")
        draft    = chain.invoke({
            "context":  context,
            "question": state["question"],
            "feedback": feedback,
        })
        logger.info(f"[Researcher] Draft produced ({len(draft)} chars).")
        return {**state, "documents": docs, "researcher_draft": draft}

    return researcher_node


def _make_direct_answer_node(llm: Any):
    """Answer without retrieval (general knowledge only)."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Answer clearly and concisely."),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()

    def direct_answer_node(state: AgentState) -> AgentState:
        answer = chain.invoke({"question": state["question"]})
        return {**state, "researcher_draft": answer, "documents": [], "critic_approved": True}

    return direct_answer_node


def _make_critic_node(llm: Any):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a critic agent. Your job is to fact-check a draft answer "
         "against the source context provided. Identify any hallucinations, "
         "inaccuracies, or unsupported claims.\n\n"
         "Source context:\n{context}\n\n"
         "Draft answer:\n{draft}\n\n"
         'Reply with JSON: {{"approved": true|false, "feedback": "..."}}'),
        ("human", "Is this draft accurate and well-supported?"),
    ])
    chain = prompt | llm | JsonOutputParser()

    def critic_node(state: AgentState) -> AgentState:
        context = "\n\n---\n\n".join(d.page_content for d in state["documents"])
        result  = chain.invoke({"context": context, "draft": state["researcher_draft"]})
        approved = result.get("approved", True)
        feedback = result.get("feedback", "")
        logger.info(f"[Critic] approved={approved}: {feedback[:100]}")
        return {
            **state,
            "critic_approved":  approved,
            "critic_feedback":  feedback,
            "iterations":       state.get("iterations", 0) + 1,
        }

    return critic_node


def _make_writer_node(llm: Any):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a writer agent. Polish the draft answer for clarity, "
         "conciseness, and readability. Remove redundancy. Keep all factual content."),
        ("human",
         "Draft:\n{draft}\n\nCritic feedback:\n{feedback}\n\n"
         "Write the final polished answer:"),
    ])
    chain = prompt | llm | StrOutputParser()

    def writer_node(state: AgentState) -> AgentState:
        final = chain.invoke({
            "draft":    state["researcher_draft"],
            "feedback": state.get("critic_feedback", "Approved."),
        })
        logger.info("[Writer] Final answer produced.")
        return {**state, "final_answer": final}

    return writer_node


# ══════════════════════════════════════════════════════════════
#  Pipeline
# ══════════════════════════════════════════════════════════════

class MultiAgentPipeline:
    """LangGraph multi-agent pipeline: Supervisor → Researcher → Critic → Writer."""

    def __init__(self) -> None:
        self.embeddings  = build_embeddings()
        self.llm         = build_llm()
        self.vectorstore = _build_vectorstore(self.embeddings)
        self.retriever   = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": TOP_K, "fetch_k": TOP_K * 3},
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        self.memory = MemorySaver()
        self.graph  = self._build_graph()
        logger.info("MultiAgentPipeline initialized.")

    def _build_graph(self):
        supervisor    = _make_supervisor_node(self.llm)
        researcher    = _make_researcher_node(self.llm, self.retriever)
        direct_answer = _make_direct_answer_node(self.llm)
        critic        = _make_critic_node(self.llm)
        writer        = _make_writer_node(self.llm)

        def route_supervisor(state: AgentState) -> Literal["researcher", "direct_answer"]:
            return "researcher" if state["retrieval_needed"] else "direct_answer"

        def route_critic(state: AgentState) -> Literal["writer", "researcher"]:
            if state["critic_approved"] or state.get("iterations", 0) >= MAX_ITERATIONS:
                return "writer"
            return "researcher"

        wf = StateGraph(AgentState)
        wf.add_node("supervisor",    supervisor)
        wf.add_node("researcher",    researcher)
        wf.add_node("direct_answer", direct_answer)
        wf.add_node("critic",        critic)
        wf.add_node("writer",        writer)

        wf.add_edge(START, "supervisor")
        wf.add_conditional_edges(
            "supervisor", route_supervisor,
            {"researcher": "researcher", "direct_answer": "direct_answer"},
        )
        wf.add_edge("researcher",    "critic")
        wf.add_edge("direct_answer", "writer")
        wf.add_conditional_edges(
            "critic", route_critic,
            {"writer": "writer", "researcher": "researcher"},
        )
        wf.add_edge("writer", END)

        return wf.compile(checkpointer=self.memory)

    # ── Ingest ────────────────────────────────────────────────
    def ingest(self) -> dict[str, Any]:
        docs = _load_documents()
        if not docs:
            return {"indexed": 0, "message": "No documents found in ./data"}
        chunks = self.splitter.split_documents(docs)
        self.vectorstore.add_documents(chunks)
        logger.info(f"Indexed {len(chunks)} chunks from {len(docs)} documents.")
        return {"indexed": len(chunks), "source_files": len(docs)}

    # ── Query ─────────────────────────────────────────────────
    def query(self, question: str, session_id: str = "default") -> dict[str, Any]:
        config = {"configurable": {"thread_id": session_id}}
        initial: AgentState = {
            "question":         question,
            "documents":        [],
            "researcher_draft": "",
            "critic_feedback":  "",
            "final_answer":     "",
            "retrieval_needed": True,
            "critic_approved":  False,
            "iterations":       0,
        }
        final = self.graph.invoke(initial, config=config)
        return {
            "answer":         final["final_answer"],
            "retrieval_used": final["retrieval_needed"],
            "critic_approved": final["critic_approved"],
            "iterations":     final["iterations"],
            "sources": [
                {
                    "content": d.page_content[:300],
                    "source":  d.metadata.get("source", "unknown"),
                    "page":    d.metadata.get("page"),
                }
                for d in final["documents"]
            ],
        }


# ── Singleton ─────────────────────────────────────────────────
_pipeline: MultiAgentPipeline | None = None


def get_pipeline() -> MultiAgentPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = MultiAgentPipeline()
    return _pipeline
