"""Local MCP server exposing knowledge-base retrieval."""

from __future__ import annotations

import logging
from pathlib import Path
import sys

from mcp.server import MCPServer
from pydantic import BaseModel, Field

# When launched by an MCP stdio client, Python puts ``mcp_servers`` rather
# than the repository root on sys.path.  Make the application package
# importable without relying on the caller's PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.loader import load_configured_knowledge_base
from app.retrieval.bm25 import BM25Search


class RAGSearchItem(BaseModel):
    """One knowledge-base chunk returned by the RAG MCP tool."""

    content: str
    score: float
    source: str
    chunk_index: int


class RAGSearchResult(BaseModel):
    """Stable structured response returned by ``search_knowledge``."""

    query: str
    results: list[RAGSearchItem] = Field(default_factory=list)


server = MCPServer(
    name="rag-server",
    description="Local knowledge-base retrieval server.",
    version="0.1.0",
)


@server.tool(
    name="search_knowledge",
    description="Search the local e-commerce knowledge base.",
    structured_output=True,
)
def search_knowledge(query: str, top_k: int = 5) -> RAGSearchResult:
    """Return the highest-scoring local knowledge-base chunks."""
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    results = BM25Search(load_configured_knowledge_base()).search(
        normalized_query,
        top_k=top_k,
    )
    return RAGSearchResult(query=normalized_query, results=results)


if __name__ == "__main__":
    # MCP stdio is a protocol channel; diagnostics must never go to stdout.
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    server.run(transport="stdio")
