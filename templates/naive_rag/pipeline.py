"""
Template A — Naive RAG (LangChain + Document Loaders)
======================================================
Drop your documents into /data, then call:

    POST /ingest   → indexes all documents in /data
    POST /query    → {"question": "..."} → answer + source chunks

This template uses:
  - LangChain document loaders  (PDF, CSV, Markdown, TXT)
  - RecursiveCharacterTextSplitter
  - ChromaDB  (or whichever VECTOR_DB you set in stack.yaml)
  - OpenAI embeddings + gpt-4o-mini  (or your chosen LLM)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from src.llm_factory import build_embeddings, build_llm

load_dotenv()

# ── Environment-resolved settings ────────────────────────────
# DATA_DIR: inside Docker = /app/data, locally = genai-sandbox/data
_HERE = Path(__file__).resolve().parent.parent.parent  # genai-sandbox/
DATA_DIR = Path(os.getenv("DATA_DIR", str(_HERE / "data")))
COLLECTION = os.getenv("COLLECTION_NAME", "sandbox_docs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "5"))
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))


# ── Vector store factory ──────────────────────────────────────
def _build_vectorstore(embeddings: Any) -> Any:
    vector_db = os.getenv("SANDBOX_VECTOR_DB", "chroma").lower()

    if vector_db == "chroma":
        from langchain_chroma import Chroma
        _chroma_dir = str(DATA_DIR / ".chroma")
        return Chroma(
            collection_name=COLLECTION,
            embedding_function=embeddings,
            persist_directory=_chroma_dir,
        )
    if vector_db == "qdrant":
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
        client = QdrantClient(url=os.getenv("QDRANT_URL", "http://genai-qdrant:6333"))
        return QdrantVectorStore(
            client=client,
            collection_name=COLLECTION,
            embedding=embeddings,
        )
    if vector_db == "faiss":
        from langchain_community.vectorstores import FAISS
        faiss_path = "/app/data/.faiss"
        if Path(faiss_path).exists():
            return FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        return FAISS.from_texts(["placeholder"], embeddings)

    raise ValueError(f"Unsupported vector_db: {vector_db}")


# ── Document loading ──────────────────────────────────────────
LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".csv": CSVLoader,
    ".md":  TextLoader,   # TextLoader works for markdown without unstructured
    ".txt": TextLoader,
    ".py":  TextLoader,
    ".json": TextLoader,
    ".html": TextLoader,
}


def load_documents() -> list[Document]:
    """Load all supported documents from DATA_DIR."""
    docs: list[Document] = []
    for path in DATA_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in LOADER_MAP:
            loader_cls = LOADER_MAP[path.suffix.lower()]
            try:
                loader = loader_cls(str(path))
                docs.extend(loader.load())
                logger.info(f"Loaded: {path.name}")
            except Exception as exc:
                logger.warning(f"Failed to load {path.name}: {exc}")
    logger.info(f"Total documents loaded: {len(docs)}")
    return docs


# ── Main RAG pipeline ─────────────────────────────────────────
class NaiveRAG:
    """Simple retrieval-augmented generation pipeline."""

    SYSTEM_PROMPT = (
        "You are a helpful AI assistant. Use only the provided context to "
        "answer the question. If the context does not contain enough "
        "information, say so clearly. Be concise and accurate.\n\n"
        "Context:\n{context}"
    )

    def __init__(self) -> None:
        self.embeddings = build_embeddings()
        self.llm = build_llm()
        self.vectorstore = _build_vectorstore(self.embeddings)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        logger.info("NaiveRAG pipeline initialized.")

    # ── Ingest ────────────────────────────────────────────────
    def ingest(self) -> dict[str, int]:
        raw_docs = load_documents()
        if not raw_docs:
            return {"indexed": 0, "message": "No documents found in /data."}
        chunks = self.splitter.split_documents(raw_docs)
        self.vectorstore.add_documents(chunks)
        logger.info(f"Indexed {len(chunks)} chunks from {len(raw_docs)} documents.")
        return {"indexed": len(chunks), "source_files": len(raw_docs)}

    # ── Query ─────────────────────────────────────────────────
    def query(self, question: str, session_id: str = "default") -> dict[str, Any]:
        from langchain_core.messages import HumanMessage, SystemMessage

        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": TOP_K, "fetch_k": TOP_K * 3},
        )
        source_docs = retriever.invoke(question)
        context = "\n\n---\n\n".join(d.page_content for d in source_docs)

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT.format(context=context)),
            HumanMessage(content=question),
        ]
        response = self.llm.invoke(messages)

        return {
            "answer": response.content,
            "sources": [
                {
                    "content": d.page_content[:300],
                    "source": d.metadata.get("source", "unknown"),
                    "page": d.metadata.get("page", None),
                }
                for d in source_docs
            ],
        }


# ── Singleton pipeline (shared across FastAPI requests) ───────
_pipeline: NaiveRAG | None = None


def get_pipeline() -> NaiveRAG:
    global _pipeline
    if _pipeline is None:
        _pipeline = NaiveRAG()
    return _pipeline
