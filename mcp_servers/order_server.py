"""Local MCP server exposing the project's mock order lookup tool."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.item_service import ItemService
from mcp.server import MCPServer
from pydantic import BaseModel, Field


MOCK_ORDERS: dict[str, dict[str, str | None]] = {
    "TEST1001": {
        "order_status": "已发货",
        "logistics_status": "运输中",
        "tracking_no": "MOCKEXP1001",
    },
    "TEST1002": {
        "order_status": "待发货",
        "logistics_status": "暂无物流",
        "tracking_no": None,
    },
    "TEST1003": {
        "order_status": "已签收",
        "logistics_status": "已签收",
        "tracking_no": "MOCKEXP1003",
    },
}


class OrderResult(BaseModel):
    """Stable structured response returned by ``get_order``."""

    found: bool
    order_id: str
    order_status: str | None = None
    logistics_status: str | None = None
    tracking_no: str | None = None


server = MCPServer(
    name="order-server",
    description="Local mock order lookup server.",
    version="0.1.0",
)


@server.tool(
    name="get_order",
    description="Look up one local mock order by order ID.",
    structured_output=True,
)
def get_order(order_id: str) -> OrderResult:
    """Return one mock order, or a non-error not-found response."""

    normalized_id = order_id.strip().upper()
    order = MOCK_ORDERS.get(normalized_id)
    if order is None:
        return OrderResult(found=False, order_id=normalized_id)

    return OrderResult(found=True, order_id=normalized_id, **order)


class ItemResult(BaseModel):
    """Public readonly item fields returned by ``get_item_info``."""

    found: bool
    item_id: str | None = None
    title: str | None = None
    listed_price_cents: int | None = None
    sale_status: str | None = None
    data_source: str | None = None
    updated_at: str | None = None
    facts: dict[str, Any] = Field(default_factory=dict)


@server.tool(
    name="get_item_info",
    description="Look up public information for one local Xianyu demo item.",
    structured_output=True,
)
def get_item_info(item_id: str) -> ItemResult:
    """Return public item facts only; no internal pricing or order actions."""

    result = ItemService().get_item_info(item_id)
    return ItemResult(**result)


if __name__ == "__main__":
    # MCP stdio is a protocol channel; diagnostics must never go to stdout.
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    server.run(transport="stdio")
