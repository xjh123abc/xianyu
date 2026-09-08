"""MCP client for the local knowledge-base retrieval server."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any

from mcp import Client, StdioServerParameters


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAG_SERVER_PATH = PROJECT_ROOT / "mcp_servers" / "rag_server.py"


class RAGMCPError(RuntimeError):
    """Raised when the RAG MCP server or tool call fails."""


def _server_parameters() -> StdioServerParameters:
    """Build stdio parameters using the active Python interpreter."""
    if not RAG_SERVER_PATH.is_file():
        raise FileNotFoundError(f"RAG MCP server entry point not found: {RAG_SERVER_PATH}")
    return StdioServerParameters(
        command=sys.executable,
        args=[str(RAG_SERVER_PATH)],
        cwd=str(PROJECT_ROOT),
        encoding="utf-8",
    )


def _format_tool_error(result: Any) -> str:
    messages = [
        block.text
        for block in getattr(result, "content", [])
        if getattr(block, "text", None)
    ]
    return "; ".join(messages) or "RAG MCP tool returned an error"


def _extract_search_result(result: Any) -> dict[str, Any]:
    if result.is_error:
        raise RAGMCPError(_format_tool_error(result))

    structured_content = result.structured_content
    if not isinstance(structured_content, Mapping):
        raise RAGMCPError("RAG MCP search returned no structured content")
    if "query" not in structured_content or "results" not in structured_content:
        raise RAGMCPError("RAG MCP search response is missing fields")

    return {
        "query": str(structured_content["query"]),
        "results": list(structured_content["results"]),
    }


async def _call_search_tool(
    client: Client,
    query: str,
    top_k: int = 5,
) -> dict[str, Any]:
    result = await client.call_tool(
        "search_knowledge",
        {"query": query, "top_k": top_k},
        read_timeout_seconds=10,
    )
    return _extract_search_result(result)


async def search_knowledge_via_mcp(
    query: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """Start the local RAG MCP server, search, and close the connection."""
    try:
        async with Client(_server_parameters(), read_timeout_seconds=10) as client:
            return await _call_search_tool(client, query, top_k)
    except RAGMCPError:
        raise
    except Exception as exc:
        raise RAGMCPError(f"RAG MCP search failed: {exc}") from exc
