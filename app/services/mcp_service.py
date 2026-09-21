"""MCP application service for order queries."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from app.infrastructure.order_mcp_client import get_item_info_via_mcp, get_order_via_mcp


OrderLookup = Callable[[str], Awaitable[dict[str, Any]]]
ItemLookup = Callable[[str], Awaitable[dict[str, Any]]]


class MCPService:
    """Keep MCP transport details outside ChatService orchestration."""

    def __init__(
        self,
        order_lookup: OrderLookup | None = None,
        item_lookup: ItemLookup | None = None,
    ) -> None:
        self._order_lookup = order_lookup or get_order_via_mcp
        self._item_lookup = item_lookup or get_item_info_via_mcp

    async def get_order(self, order_id: str) -> dict[str, Any]:
        """Fetch one normalized order ID through MCP."""
        normalized_id = order_id.strip().upper()
        if not normalized_id:
            raise ValueError("order_id must not be empty")
        return await self._order_lookup(normalized_id)

    async def get_item_info(self, item_id: str) -> dict[str, Any]:
        """Fetch one public Xianyu item record through MCP."""
        normalized_id = item_id.strip().upper()
        if not normalized_id:
            raise ValueError("item_id must not be empty")
        return await self._item_lookup(normalized_id)
