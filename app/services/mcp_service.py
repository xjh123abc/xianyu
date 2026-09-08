"""Application service for MCP-backed business data."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from app.infrastructure.order_mcp_client import get_order_via_mcp
from app.infrastructure.rag_mcp_client import search_knowledge_via_mcp


OrderLookup = Callable[[str], Awaitable[dict[str, Any]]]
RAGLookup = Callable[[str, int], Awaitable[dict[str, Any]]]


class MCPService:
    """Keep MCP transport details out of the chat orchestration layer."""

    def __init__(
        self,
        order_lookup: OrderLookup | None = None,
        rag_lookup: RAGLookup | None = None,
    ) -> None:
        self._order_lookup = order_lookup or get_order_via_mcp
        self._rag_lookup = rag_lookup or search_knowledge_via_mcp

    async def get_order(self, order_id: str) -> dict[str, Any]:
        """Fetch one order through the configured MCP client."""
        normalized_id = order_id.strip().upper()
        if not normalized_id:
            raise ValueError("order_id must not be empty")
        return await self._order_lookup(normalized_id)

    async def run(self, order_id: str | None) -> dict[str, Any]:
        """Return a structured MCP result suitable for a combined prompt."""
        if order_id is None or not order_id.strip():
            return {
                "found": False,
                "order_id": None,
                "error": "missing_order_id",
            }
        return await self.get_order(order_id)

    async def search_knowledge(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """Expose the RAG MCP capability through the same service boundary."""
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")
        return await self._rag_lookup(normalized_query, top_k)
