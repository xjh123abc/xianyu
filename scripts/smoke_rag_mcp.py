"""End-to-end smoke test for the local RAG MCP server."""

from __future__ import annotations

import asyncio
import json
import sys

from mcp import Client

from app.infrastructure.rag_mcp_client import _call_search_tool, _server_parameters


async def run_smoke_test() -> None:
    """Exercise tools/list and tools/call through a real subprocess."""
    async with Client(_server_parameters(), read_timeout_seconds=10) as client:
        tool_list = await client.list_tools()
        tool_names = [tool.name for tool in tool_list.tools]
        if tool_names != ["search_knowledge"]:
            raise AssertionError(f"Unexpected RAG MCP tools: {tool_names}")
        print(json.dumps({"discovered_tools": tool_names}, ensure_ascii=False))

        result = await _call_search_tool(client, "退货", top_k=3)
        if result["query"] != "退货" or len(result["results"]) > 3:
            raise AssertionError(f"Unexpected RAG MCP response: {result!r}")
        print(json.dumps(result, ensure_ascii=False))


def main() -> None:
    try:
        asyncio.run(run_smoke_test())
    except Exception as exc:
        print(f"RAG MCP smoke test failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
