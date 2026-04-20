"""
Template B — Agentic RAG (LangGraph 1.x)
=========================================
A full agentic loop with:
  1. Router node  — decides whether the question needs retrieval
  2. Retrieve node — MMR retrieval from vector store
  3. Grade node   — scores retrieved chunks for relevance
  4. Generate node — produces the final answer
  5. Fallback node — web-search or "I don't know" when grading fails

Graph topology:

    [router] ──► [retrieve] ──► [grade] ──► [generate]
        │                           │
        │ (no_retrieval)            └──► [fallback]
        ▼
    [generate]

Usage (via the serving layer):
    POST /ingest   → indexes /data documents
    POST /query    → {"question": "...", "session_id": "..."}
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from src.llm_factory import build_embeddings, build_llm
from langgraph.graph import END, START, StateGraph
from loguru import logger

load_dotenv()

# ── Settings ──────────────────────────────────────────────────
DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))
COLLECTION = os.getenv("COLLECTION_NAME", "sandbox_docs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "5"))
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")


# ── Shared LLM ────────────────────────────────────────────────
_llm = build_llm()
_llm_json = build_llm()


# ── Graph State ───────────────────────────────────────────────
class GraphState(TypedDict):
    question: str
    documents: list[Document]
    generation: str
    retrieval_required: bool
    grade_passed: bool
    iterations: int


# ── Vector store factory (shared with naive_rag) ──────────────
def _build_vectorstore(embeddings: Any) -> Any:
    vdb = os.getenv("SANDBOX_VECTOR_DB", "chroma").lower()
    if vdb == "chroma":
        from langchain_chroma import Chroma
        return Chroma(
            collection_name=COLLECTION,
            embedding_function=embeddings,
            persist_directory="/app/data/.chroma",
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


# ── Document loader (reused from naive_rag) ───────────────────
def _load_raw_documents() -> list[Document]:
    from langchain_community.document_loaders import (
        CSVLoader, PyPDFLoader, TextLoader, UnstructuredMarkdownLoader,
    )
    ext_map = {
        ".pdf": PyPDFLoader,
        ".csv": CSVLoader,
        ".md":  UnstructuredMarkdownLoader,
        ".txt": TextLoader,
    }
    docs: list[Document] = []
    for p in DATA_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in ext_map:
            try:
                docs.extend(ext_map[p.suffix.lower()](str(p)).load())
            except Exception as exc:
                logger.warning(f"Skipping {p.name}: {exc}")
    return docs


# ══════════════════════════════════════════════════════════════
#  Node implementations
# ══════════════════════════════════════════════════════════════

def _router_node(state: GraphState) -> GraphState:
    """Decide whether the question requires document retrieval."""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a routing agent. Determine whether the user's question "
         "requires searching a document knowledge base or can be answered "
         "directly from general knowledge. "
         'Reply with JSON: {{"retrieval_required": true|false, "reason": "..."}}'),
        ("human", "{question}"),
    ])
    chain = prompt | _llm_json | JsonOutputParser()
    result = chain.invoke({"question": state["question"]})
    logger.info(f"Router decision: {result}")
    return {**state, "retrieval_required": result.get("retrieval_required", True)}


def _retrieve_node(state: GraphState, retriever: Any) -> GraphState:
    """Retrieve relevant document chunks from the vector store."""
    docs = retriever.invoke(state["question"])
    logger.info(f"Retrieved {len(docs)} documents.")
    return {**state, "documents": docs}


def _grade_node(state: GraphState) -> GraphState:
    """Grade retrieved documents for relevance to the question."""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a grader. Given a question and a document chunk, decide if "
         "the chunk is relevant to answering the question. "
         'Reply with JSON: {{"relevant": true|false}}'),
        ("human", "Question: {question}\n\nDocument: {document}"),
    ])
    chain = prompt | _llm_json | JsonOutputParser()

    relevant = [
        doc for doc in state["documents"]
        if chain.invoke({"question": state["question"], "document": doc.page_content})
            .get("relevant", False)
    ]
    grade_passed = len(relevant) > 0
    logger.info(f"Grading: {len(relevant)}/{len(state['documents'])} relevant.")
    return {**state, "documents": relevant, "grade_passed": grade_passed}


def _generate_node(state: GraphState) -> GraphState:
    """Generate the final answer using retrieved context."""
    context = "\n\n---\n\n".join(d.page_content for d in state["documents"])
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful AI assistant. Use the provided context to answer "
         "the question accurately and concisely. Cite sources where possible.\n\n"
         "Context:\n{context}"),
        ("human", "{question}"),
    ])
    chain = prompt | _llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": state["question"]})
    return {**state, "generation": answer, "iterations": state.get("iterations", 0) + 1}


def _direct_generate_node(state: GraphState) -> GraphState:
    """Generate an answer without document retrieval (general knowledge)."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Answer the question clearly and concisely."),
        ("human", "{question}"),
    ])
    chain = prompt | _llm | StrOutputParser()
    answer = chain.invoke({"question": state["question"]})
    return {**state, "generation": answer, "documents": []}


def _fallback_node(state: GraphState) -> GraphState:
    """Fallback when grading fails — attempt a web search or graceful decline."""
    logger.warning("Grading failed — entering fallback node.")
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        search = DuckDuckGoSearchRun()
        results = search.run(state["question"])
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant. Use the web search results below "
             "to answer the question. If they are insufficient, say so.\n\n"
             "Search results:\n{results}"),
            ("human", "{question}"),
        ])
        chain = prompt | _llm | StrOutputParser()
        answer = chain.invoke({"results": results, "question": state["question"]})
    except Exception:
        answer = (
            "I could not find sufficient information in the knowledge base or "
            "via web search to answer this question accurately."
        )
    return {**state, "generation": answer, "documents": []}


# ── Edge routing functions ────────────────────────────────────
def _route_after_router(state: GraphState) -> Literal["retrieve", "direct_generate"]:
    return "retrieve" if state["retrieval_required"] else "direct_generate"


def _route_after_grade(state: GraphState) -> Literal["generate", "fallback"]:
    return "generate" if state["grade_passed"] else "fallback"


# ══════════════════════════════════════════════════════════════
#  Graph builder
# ══════════════════════════════════════════════════════════════

class AgenticRAG:
    """LangGraph-powered agentic RAG pipeline."""

    def __init__(self) -> None:
        self.embeddings = build_embeddings()
        self.vectorstore = _build_vectorstore(self.embeddings)
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": TOP_K, "fetch_k": TOP_K * 3},
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        self.memory = MemorySaver()
        self.graph = self._build_graph()
        logger.info("AgenticRAG pipeline initialized.")

    def _build_graph(self):
        retriever = self.retriever  # capture for closure

        def retrieve(state: GraphState) -> GraphState:
            return _retrieve_node(state, retriever)

        workflow = StateGraph(GraphState)

        # ── Add nodes ──────────────────────────────────────────
        workflow.add_node("router",          _router_node)
        workflow.add_node("retrieve",        retrieve)
        workflow.add_node("grade",           _grade_node)
        workflow.add_node("generate",        _generate_node)
        workflow.add_node("direct_generate", _direct_generate_node)
        workflow.add_node("fallback",        _fallback_node)

        # ── Add edges ──────────────────────────────────────────
        workflow.add_edge(START, "router")
        workflow.add_conditional_edges(
            "router",
            _route_after_router,
            {"retrieve": "retrieve", "direct_generate": "direct_generate"},
        )
        workflow.add_edge("retrieve", "grade")
        workflow.add_conditional_edges(
            "grade",
            _route_after_grade,
            {"generate": "generate", "fallback": "fallback"},
        )
        workflow.add_edge("generate",        END)
        workflow.add_edge("direct_generate", END)
        workflow.add_edge("fallback",        END)

        return workflow.compile(checkpointer=self.memory)

    # ── Public methods ────────────────────────────────────────
    def ingest(self) -> dict[str, int]:
        raw = _load_raw_documents()
        if not raw:
            return {"indexed": 0, "message": "No documents found in /data."}
        chunks = self.splitter.split_documents(raw)
        self.vectorstore.add_documents(chunks)
        logger.info(f"Indexed {len(chunks)} chunks from {len(raw)} documents.")
        return {"indexed": len(chunks), "source_files": len(raw)}

    def query(self, question: str, session_id: str = "default") -> dict[str, Any]:
        config = {"configurable": {"thread_id": session_id}}
        initial_state: GraphState = {
            "question": question,
            "documents": [],
            "generation": "",
            "retrieval_required": True,
            "grade_passed": False,
            "iterations": 0,
        }
        final_state = self.graph.invoke(initial_state, config=config)
        return {
            "answer": final_state["generation"],
            "retrieval_used": final_state["retrieval_required"],
            "grade_passed": final_state.get("grade_passed", False),
            "sources": [
                {
                    "content": d.page_content[:300],
                    "source": d.metadata.get("source", "unknown"),
                }
                for d in final_state["documents"]
            ],
        }


# ── Singleton ─────────────────────────────────────────────────
_pipeline: AgenticRAG | None = None


def get_pipeline() -> AgenticRAG:
    global _pipeline
    if _pipeline is None:
        _pipeline = AgenticRAG()
    return _pipeline
