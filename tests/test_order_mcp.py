"""Focused tests for the isolated mock order MCP boundary."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest

from app.infrastructure import order_mcp_client
from mcp_servers.order_server import OrderResult, get_order, server


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


def test_server_discovers_only_get_order() -> None:
    tool_list = asyncio.run(server.list_tools())

    assert [tool.name for tool in tool_list] == ["get_order"]


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
