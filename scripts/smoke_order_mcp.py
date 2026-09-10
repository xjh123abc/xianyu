"""End-to-end smoke test for the order MCP server and client transport."""

from __future__ import annotations

import asyncio
import json
import sys

from mcp import Client

from app.infrastructure.order_mcp_client import (
    _call_item_tool,
    _call_order_tool,
    _server_parameters,
)


EXPECTED_ORDERS = {
    "TEST1001": {
        "found": True,
        "order_id": "TEST1001",
        "order_status": "已发货",
        "logistics_status": "运输中",
        "tracking_no": "MOCKEXP1001",
    },
    "TEST1002": {
        "found": True,
        "order_id": "TEST1002",
        "order_status": "待发货",
        "logistics_status": "暂无物流",
        "tracking_no": None,
    },
    "TEST1003": {
        "found": True,
        "order_id": "TEST1003",
        "order_status": "已签收",
        "logistics_status": "已签收",
        "tracking_no": "MOCKEXP1003",
    },
}

EXPECTED_MISSING_ORDER = {
    "found": False,
    "order_id": "TEST9999",
    "order_status": None,
    "logistics_status": None,
    "tracking_no": None,
}

EXPECTED_ITEMS = {
    "DEMO_ITEM_001": {
        "found": True,
        "listed_price_cents": 128000,
        "sale_status": "listed",
    },
    "DEMO_ITEM_002": {
        "found": True,
        "listed_price_cents": 56000,
        "sale_status": "sold",
    },
    "DEMO_ITEM_003": {
        "found": True,
        "listed_price_cents": 98000,
        "sale_status": "unknown",
    },
}


async def run_smoke_test() -> None:
    """Exercise tools/list and tools/call through a real subprocess."""

    async with Client(_server_parameters(), read_timeout_seconds=10) as client:
        tool_list = await client.list_tools()
        tool_names = [tool.name for tool in tool_list.tools]
        if tool_names != ["get_order", "get_item_info"]:
            raise AssertionError(f"Unexpected MCP tools: {tool_names}")
        print(json.dumps({"discovered_tools": tool_names}, ensure_ascii=False))

        for order_id, expected in EXPECTED_ORDERS.items():
            actual = await _call_order_tool(client, order_id)
            if actual != expected:
                raise AssertionError(
                    f"Unexpected response for {order_id}: {actual!r} != {expected!r}"
                )
            print(json.dumps(actual, ensure_ascii=False))

        actual_missing = await _call_order_tool(client, "TEST9999")
        if actual_missing != EXPECTED_MISSING_ORDER:
            raise AssertionError(
                "Unexpected not-found response: "
                f"{actual_missing!r} != {EXPECTED_MISSING_ORDER!r}"
            )
        print(json.dumps(actual_missing, ensure_ascii=False))

        for item_id, expected_fields in EXPECTED_ITEMS.items():
            item = await _call_item_tool(client, item_id)
            if item.get("item_id") != item_id or any(
                item.get(key) != value for key, value in expected_fields.items()
            ):
                raise AssertionError(f"Unexpected item response for {item_id}: {item!r}")
            print(json.dumps(item, ensure_ascii=False))

        missing_item = await _call_item_tool(client, "XXX999")
        if (
            missing_item.get("found") is not False
            or missing_item.get("item_id") != "XXX999"
        ):
            raise AssertionError(f"Unexpected missing item response: {missing_item!r}")
        print(json.dumps(missing_item, ensure_ascii=False))


def main() -> None:
    try:
        asyncio.run(run_smoke_test())
    except Exception as exc:
        print(f"MCP smoke test failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
