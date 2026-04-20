"""
Template D — Structured Output (PydanticAI + Tool Calling)
===========================================================
Demonstrates two 2026-era patterns:

  1. Structured extraction — ask for a Pydantic model back (not free text)
  2. Tool-augmented RAG   — the LLM decides when to search vs answer directly

GraphState adds a `structured` key that holds the Pydantic model when
the caller requests typed output.

Usage:
    POST /ingest  → index documents
    POST /query   → {"question": "...", "output_schema": "summary|qa|entities"}

Output schemas:
    summary   → SummaryOutput  (title, bullets, confidence)
    qa        → QAOutput       (answer, sources, confidence, follow_up_questions)
    entities  → EntityOutput   (entities with type + description)
    (default) → plain string answer

Run standalone:
    from templates.structured_output.pipeline import get_pipeline
    p = get_pipeline()
    p.ingest()
    result = p.query("What are the main findings?", output_schema="qa")
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(HERE))
load_dotenv(HERE / ".env")

from src.llm_factory import build_embeddings, build_llm  # noqa: E402

# ── Settings ──────────────────────────────────────────────────
DATA_DIR      = Path(os.getenv("DATA_DIR", str(HERE / "data")))
COLLECTION    = os.getenv("COLLECTION_NAME", "sandbox_docs")
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K         = int(os.getenv("TOP_K", "5"))


# ══════════════════════════════════════════════════════════════
#  Output Schemas
# ══════════════════════════════════════════════════════════════

class QAOutput(BaseModel):
    """Structured question-answer with confidence and follow-ups."""
    answer:              str         = Field(description="Direct answer to the question.")
    confidence:          float       = Field(ge=0.0, le=1.0, description="Confidence 0-1.")
    sources_used:        list[str]   = Field(default_factory=list, description="Source file names.")
    follow_up_questions: list[str]   = Field(default_factory=list, description="Suggested follow-up questions.")


class SummaryOutput(BaseModel):
    """Structured document summary."""
    title:       str       = Field(description="Short descriptive title.")
    bullets:     list[str] = Field(description="Key points as bullet strings.")
    confidence:  float     = Field(ge=0.0, le=1.0)
    word_count:  int       = Field(description="Approximate word count of source material.")


class EntityOutput(BaseModel):
    """Named entities extracted from the document context."""

    class Entity(BaseModel):
        name:        str = Field(description="Entity name.")
        entity_type: str = Field(description="Type: PERSON | ORG | CONCEPT | TECHNIQUE | DATE | LOCATION")
        description: str = Field(description="One-sentence description of this entity in context.")

    entities:   list[Entity] = Field(description="All extracted entities.")
    confidence: float        = Field(ge=0.0, le=1.0)


SCHEMA_MAP: dict[str, type[BaseModel]] = {
    "qa":       QAOutput,
    "summary":  SummaryOutput,
    "entities": EntityOutput,
}


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
    if vdb == "faiss":
        from langchain_community.vectorstores import FAISS
        faiss_path = str(DATA_DIR / ".faiss")
        if Path(faiss_path).exists():
            return FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        return FAISS.from_texts(["placeholder"], embeddings)
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
#  Pipeline
# ══════════════════════════════════════════════════════════════

class StructuredOutputPipeline:
    """
    RAG pipeline that returns typed Pydantic models instead of free text.
    Uses LangChain's `.with_structured_output()` when a schema is requested.
    """

    def __init__(self) -> None:
        self.embeddings  = build_embeddings()
        self.llm         = build_llm()
        self.vectorstore = _build_vectorstore(self.embeddings)
        self.splitter    = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        logger.info("StructuredOutputPipeline initialized.")

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
    def query(
        self,
        question: str,
        session_id: str = "default",
        output_schema: str = "qa",
    ) -> dict[str, Any]:
        """
        Query the knowledge base and return structured output.

        Args:
            question:      The user question.
            session_id:    Unused (present for API compatibility).
            output_schema: "qa" | "summary" | "entities" | "" for plain text.

        Returns:
            dict with "answer" (str or dict) and "sources".
        """
        retriever  = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": TOP_K, "fetch_k": TOP_K * 3},
        )
        source_docs = retriever.invoke(question)
        context     = "\n\n---\n\n".join(d.page_content for d in source_docs)
        source_names = list({
            Path(d.metadata.get("source", "unknown")).name for d in source_docs
        })

        schema_cls = SCHEMA_MAP.get(output_schema.lower()) if output_schema else None

        if schema_cls is not None:
            # ── Structured output path ─────────────────────────
            structured_llm = self.llm.with_structured_output(schema_cls)
            prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are a precise AI assistant. Use only the provided context. "
                 "Return a structured response that exactly matches the requested schema.\n\n"
                 "Context:\n{context}"),
                ("human", "{question}"),
            ])
            chain  = prompt | structured_llm
            result = chain.invoke({"context": context, "question": question})

            # Inject sources if the schema has the field
            answer_dict = result.model_dump()
            if "sources_used" in answer_dict and not answer_dict["sources_used"]:
                answer_dict["sources_used"] = source_names

            return {
                "answer":       answer_dict,
                "output_schema": output_schema,
                "sources": [
                    {
                        "content": d.page_content[:300],
                        "source":  d.metadata.get("source", "unknown"),
                        "page":    d.metadata.get("page"),
                    }
                    for d in source_docs
                ],
            }
        else:
            # ── Plain text fallback ────────────────────────────
            from langchain_core.messages import HumanMessage, SystemMessage
            messages = [
                SystemMessage(content=(
                    "You are a helpful AI assistant. Use only the provided context.\n\n"
                    f"Context:\n{context}"
                )),
                HumanMessage(content=question),
            ]
            response = self.llm.invoke(messages)
            return {
                "answer": response.content,
                "output_schema": "text",
                "sources": [
                    {
                        "content": d.page_content[:300],
                        "source":  d.metadata.get("source", "unknown"),
                        "page":    d.metadata.get("page"),
                    }
                    for d in source_docs
                ],
            }


# ── Singleton ─────────────────────────────────────────────────
_pipeline: StructuredOutputPipeline | None = None


def get_pipeline() -> StructuredOutputPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = StructuredOutputPipeline()
    return _pipeline
