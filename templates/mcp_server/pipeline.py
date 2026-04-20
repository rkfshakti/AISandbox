"""
Template C — MCP Server (Model Context Protocol)
=================================================
Exposes the indexed knowledge base as MCP tools so any MCP-compatible
AI client (GitHub Copilot, Claude Desktop, Cursor, Windsurf, etc.)
can query your documents directly.

MCP tools exposed:
  • search_knowledge_base  — semantic search over ingested documents
  • ingest_documents       — trigger document ingestion
  • get_sandbox_status     — report current stack config

Run standalone:
    python3 -m templates.mcp_server.pipeline
    — or —
    make serve-mcp          (starts on stdio transport, default for MCP)

Claude Desktop config (~/.config/claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "genai-sandbox": {
          "command": "python3",
          "args": ["-m", "templates.mcp_server.pipeline"],
          "cwd": "/path/to/genai-sandbox"
        }
      }
    }

GitHub Copilot / VS Code config (.vscode/mcp.json):
    {
      "servers": {
        "genai-sandbox": {
          "type": "stdio",
          "command": "python3",
          "args": ["-m", "templates.mcp_server.pipeline"],
          "cwd": "${workspaceFolder}"
        }
      }
    }
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger

# ── Path bootstrap ─────────────────────────────────────────────
HERE = Path(__file__).resolve().parent.parent.parent  # genai-sandbox/
sys.path.insert(0, str(HERE))
load_dotenv(HERE / ".env")

DATA_DIR   = Path(os.getenv("DATA_DIR", str(HERE / "data")))
TOP_K      = int(os.getenv("TOP_K", "5"))
COLLECTION = os.getenv("COLLECTION_NAME", "sandbox_docs")

# ── Lazy vector store ──────────────────────────────────────────
_vectorstore: Any = None


def _get_vectorstore() -> Any:
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore
    from src.llm_factory import build_embeddings
    from langchain_chroma import Chroma
    _vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=build_embeddings(),
        persist_directory=str(DATA_DIR / ".chroma"),
    )
    return _vectorstore


# ══════════════════════════════════════════════════════════════
#  MCP Tool implementations
# ══════════════════════════════════════════════════════════════

def search_knowledge_base(query: str, top_k: int = TOP_K) -> dict:
    """
    Perform semantic search over all ingested documents.

    Args:
        query: The search query or question.
        top_k: Number of results to return (default 5).

    Returns:
        A dict with 'results' (list of chunks with source + content).
    """
    try:
        vs = _get_vectorstore()
        retriever = vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k, "fetch_k": top_k * 3},
        )
        docs = retriever.invoke(query)
        return {
            "results": [
                {
                    "content": d.page_content,
                    "source":  d.metadata.get("source", "unknown"),
                    "page":    d.metadata.get("page"),
                }
                for d in docs
            ],
            "total": len(docs),
        }
    except Exception as exc:
        logger.error(f"search_knowledge_base failed: {exc}")
        return {"error": str(exc), "results": []}


def ingest_documents() -> dict:
    """
    Scan ./data and index all supported documents (PDF, CSV, MD, TXT).

    Returns:
        A dict with 'indexed' chunk count and 'source_files' count.
    """
    try:
        from langchain_community.document_loaders import (
            CSVLoader, PyPDFLoader, TextLoader,
        )
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        LOADER_MAP = {
            ".pdf": PyPDFLoader,
            ".csv": CSVLoader,
            ".md":  TextLoader,
            ".txt": TextLoader,
            ".py":  TextLoader,
            ".json": TextLoader,
        }
        chunk_size    = int(os.getenv("CHUNK_SIZE", "1000"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "150"))

        raw_docs = []
        for path in DATA_DIR.rglob("*"):
            if path.is_file() and path.suffix.lower() in LOADER_MAP:
                try:
                    raw_docs.extend(LOADER_MAP[path.suffix.lower()](str(path)).load())
                except Exception as exc:
                    logger.warning(f"Skipping {path.name}: {exc}")

        if not raw_docs:
            return {"indexed": 0, "message": "No documents found in ./data"}

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(raw_docs)
        vs = _get_vectorstore()
        vs.add_documents(chunks)
        return {"indexed": len(chunks), "source_files": len(raw_docs)}
    except Exception as exc:
        logger.error(f"ingest_documents failed: {exc}")
        return {"error": str(exc)}


def get_sandbox_status() -> dict:
    """Return current sandbox configuration and health status."""
    return {
        "status":       "ok",
        "template":     "mcp_server",
        "llm_provider": os.getenv("LLM_PROVIDER", "openai"),
        "vector_db":    os.getenv("SANDBOX_VECTOR_DB", "chroma"),
        "chat_model":   os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        "embed_model":  os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        "data_dir":     str(DATA_DIR),
        "collection":   COLLECTION,
    }


# ══════════════════════════════════════════════════════════════
#  MCP Protocol — stdio transport (JSON-RPC 2.0)
# ══════════════════════════════════════════════════════════════

TOOL_REGISTRY = {
    "search_knowledge_base": {
        "fn":          search_knowledge_base,
        "description": "Semantic search over all ingested documents in the sandbox knowledge base.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type":        "string",
                    "description": "The search query or question.",
                },
                "top_k": {
                    "type":        "integer",
                    "description": "Number of results to return (default 5).",
                    "default":     5,
                },
            },
            "required": ["query"],
        },
    },
    "ingest_documents": {
        "fn":          ingest_documents,
        "description": "Index all documents in the ./data folder into the vector store.",
        "inputSchema": {
            "type":       "object",
            "properties": {},
            "required":   [],
        },
    },
    "get_sandbox_status": {
        "fn":          get_sandbox_status,
        "description": "Return the current sandbox configuration and health status.",
        "inputSchema": {
            "type":       "object",
            "properties": {},
            "required":   [],
        },
    },
}


def _send(response: dict) -> None:
    """Write a JSON-RPC response to stdout."""
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()


def _handle_request(request: dict) -> dict | None:
    method = request.get("method", "")
    req_id = request.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id":      req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities":    {"tools": {}},
                "serverInfo": {
                    "name":    "genai-sandbox",
                    "version": "1.0.0",
                },
            },
        }

    if method == "tools/list":
        tools = [
            {
                "name":        name,
                "description": meta["description"],
                "inputSchema": meta["inputSchema"],
            }
            for name, meta in TOOL_REGISTRY.items()
        ]
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools}}

    if method == "tools/call":
        params    = request.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        if tool_name not in TOOL_REGISTRY:
            return {
                "jsonrpc": "2.0",
                "id":      req_id,
                "error":   {"code": -32601, "message": f"Tool not found: {tool_name}"},
            }

        try:
            result = TOOL_REGISTRY[tool_name]["fn"](**arguments)
            return {
                "jsonrpc": "2.0",
                "id":      req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
                    "isError": False,
                },
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id":      req_id,
                "result": {
                    "content": [{"type": "text", "text": f"Error: {exc}"}],
                    "isError": True,
                },
            }

    if method == "notifications/initialized":
        return None  # no response for notifications

    # Unknown method
    return {
        "jsonrpc": "2.0",
        "id":      req_id,
        "error":   {"code": -32601, "message": f"Method not found: {method}"},
    }


def run_stdio_server() -> None:
    """Main loop: read JSON-RPC requests from stdin, write responses to stdout."""
    logger.remove()  # silence loguru on stdout — MCP uses stdout for protocol
    logger.add(sys.stderr, level="INFO")
    logger.info("GenAI Sandbox MCP server started (stdio transport)")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request  = json.loads(line)
            response = _handle_request(request)
            if response is not None:
                _send(response)
        except json.JSONDecodeError as exc:
            _send({
                "jsonrpc": "2.0",
                "id":      None,
                "error":   {"code": -32700, "message": f"Parse error: {exc}"},
            })
        except Exception as exc:
            logger.exception(f"Unhandled error: {exc}")


if __name__ == "__main__":
    run_stdio_server()
