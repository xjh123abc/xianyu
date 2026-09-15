"""Focused tests for the isolated mock order MCP boundary."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest

from app.infrastructure import order_mcp_client
from mcp_servers.order_server import ItemResult, OrderResult, get_item_info, get_order, server


def test_get_order_returns_each_mock_order() -> None:
    result = get_order(" test1001 ")

    assert result == OrderResult(
        found=True,
        order_id="TEST1001",
        order_status="已发货",
        logistics_status="运输中",
        tracking_no="MOCKEXP1001",
    )


def test_get_order_returns_structured_not_found_response() -> None:
    result = get_order("TEST9999")

    assert result.model_dump() == {
        "found": False,
        "order_id": "TEST9999",
        "order_status": None,
        "logistics_status": None,
        "tracking_no": None,
    }


def test_get_item_info_returns_public_fields_only() -> None:
    result = get_item_info("demo_item_001")

    assert isinstance(result, ItemResult)
    assert result.found is True
    assert result.item_id == "DEMO_ITEM_001"
    assert "internal" not in result.model_dump()
    assert result.listed_price_cents == 128000


def test_client_extracts_structured_item_result() -> None:
    result = SimpleNamespace(
        is_error=False,
        structured_content={
            "found": True,
            "item_id": "DEMO_ITEM_001",
            "title": "camera",
            "listed_price_cents": 12800000,
            "sale_status": "listed",
            "data_source": "seller_manual",
            "updated_at": "2026-09-09T09:00:00+08:00",
            "facts": {},
        },
        content=[],
    )

    assert order_mcp_client._extract_item_result(result) == result.structured_content


def test_server_discovers_order_and_readonly_item_tools() -> None:
    tool_list = asyncio.run(server.list_tools())

    assert [tool.name for tool in tool_list] == ["get_order", "get_item_info"]


def test_client_starts_server_with_active_python() -> None:
    parameters = order_mcp_client._server_parameters()

    assert parameters.command == sys.executable
    assert parameters.args == [str(order_mcp_client.ORDER_SERVER_PATH)]
    assert parameters.cwd == str(order_mcp_client.PROJECT_ROOT)


def test_client_extracts_structured_result() -> None:
    result = SimpleNamespace(
        is_error=False,
        structured_content={
            "found": True,
            "order_id": "TEST1003",
            "order_status": "已签收",
            "logistics_status": "已签收",
            "tracking_no": "MOCKEXP1003",
        },
        content=[],
    )

    assert order_mcp_client._extract_order_result(result) == result.structured_content


def test_client_raises_for_mcp_error() -> None:
    result = SimpleNamespace(
        is_error=True,
        structured_content=None,
        content=[SimpleNamespace(text="server unavailable")],
    )

    with pytest.raises(order_mcp_client.OrderMCPError, match="server unavailable"):
        order_mcp_client._extract_order_result(result)
