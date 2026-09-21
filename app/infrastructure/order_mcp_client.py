"""MCP client for the local mock order server.

This module deliberately owns only the MCP transport boundary.  It does not
import the FastAPI application or any RAG/model infrastructure.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any

from mcp import Client, StdioServerParameters


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORDER_SERVER_PATH = PROJECT_ROOT / "mcp_servers" / "order_server.py"
ORDER_RESULT_KEYS = (
    "found",
    "order_id",
    "order_status",
    "logistics_status",
    "tracking_no",
)
ITEM_RESULT_KEYS = (
    "found",
    "item_id",
    "title",
    "listed_price_cents",
    "sale_status",
    "data_source",
    "updated_at",
    "facts",
)


class OrderMCPError(RuntimeError):
    """Raised when the MCP server or tool call fails."""


def _server_parameters() -> StdioServerParameters:
    """Build a platform-safe stdio command using the active Python interpreter."""

    if not ORDER_SERVER_PATH.is_file():
        raise FileNotFoundError(f"MCP server entry point not found: {ORDER_SERVER_PATH}")

    return StdioServerParameters(
        command=sys.executable,
        args=[str(ORDER_SERVER_PATH)],
        cwd=str(PROJECT_ROOT),
        encoding="utf-8",
    )


def _format_tool_error(result: Any) -> str:
    """Extract human-readable text from an MCP error result when available."""

    messages = [
        block.text
        for block in getattr(result, "content", [])
        if getattr(block, "text", None)
    ]
    return "; ".join(messages) or "MCP tool returned an error"


def _extract_order_result(result: Any) -> dict[str, Any]:
    """Validate and normalize the structured result from ``tools/call``."""

    if result.is_error:
        raise OrderMCPError(_format_tool_error(result))

    structured_content = result.structured_content
    if not isinstance(structured_content, Mapping):
        raise OrderMCPError("MCP get_order returned no structured content")

    missing_keys = [key for key in ORDER_RESULT_KEYS if key not in structured_content]
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise OrderMCPError(f"MCP get_order response is missing fields: {missing}")

    return {key: structured_content[key] for key in ORDER_RESULT_KEYS}


def _extract_item_result(result: Any) -> dict[str, Any]:
    """Validate and normalize the public item response from MCP."""

    if result.is_error:
        raise OrderMCPError(_format_tool_error(result))
    structured_content = result.structured_content
    if not isinstance(structured_content, Mapping):
        raise OrderMCPError("MCP get_item_info returned no structured content")
    missing_keys = [key for key in ITEM_RESULT_KEYS if key not in structured_content]
    if missing_keys:
        raise OrderMCPError(
            "MCP get_item_info response is missing fields: " + ", ".join(missing_keys)
        )
    return {key: structured_content[key] for key in ITEM_RESULT_KEYS}


async def _call_order_tool(client: Client, order_id: str) -> dict[str, Any]:
    """Call ``get_order`` over the already-established MCP connection."""

    result = await client.call_tool(
        "get_order",
        {"order_id": order_id},
        read_timeout_seconds=10,
    )
    return _extract_order_result(result)


async def get_order_via_mcp(order_id: str) -> dict[str, Any]:
    """Start the local MCP server, call ``get_order``, and close the connection."""

    try:
        async with Client(_server_parameters(), read_timeout_seconds=10) as client:
            return await _call_order_tool(client, order_id)
    except OrderMCPError:
        raise
    except Exception as exc:
        raise OrderMCPError(f"MCP order lookup failed: {exc}") from exc


async def _call_item_tool(client: Client, item_id: str) -> dict[str, Any]:
    """Call ``get_item_info`` over an established MCP connection."""

    result = await client.call_tool(
        "get_item_info",
        {"item_id": item_id},
        read_timeout_seconds=10,
    )
    return _extract_item_result(result)


async def get_item_info_via_mcp(item_id: str) -> dict[str, Any]:
    """Start the local MCP server, call the readonly item tool, and close it."""

    try:
        async with Client(_server_parameters(), read_timeout_seconds=10) as client:
            return await _call_item_tool(client, item_id)
    except OrderMCPError:
        raise
    except Exception as exc:
        raise OrderMCPError(f"MCP item lookup failed: {exc}") from exc
