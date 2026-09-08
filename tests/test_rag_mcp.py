"""Tests for the RAG MCP protocol boundary and combined chat route."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from app.generation.deepseek import DeepSeekGenerator
from app.generation.prompt import build_combined_messages
from app.infrastructure import rag_mcp_client
from app.services.chat_service import ChatService, route_query
from mcp_servers.rag_server import RAGSearchResult, search_knowledge, server


ORDER_DATA = {
    "found": True,
    "order_id": "TEST1001",
    "order_status": "已发货",
    "logistics_status": "运输中",
    "tracking_no": "MOCKEXP1001",
}


def test_combined_route_is_selected_only_for_two_source_questions() -> None:
    assert route_query("帮我查订单 TEST1001 的物流状态，以及退货政策？") == (
        "rag_mcp",
        "TEST1001",
    )
    assert route_query("帮我查订单 TEST1001 发货了吗？") == ("order", "TEST1001")
    assert route_query("订单一般多久发货？") == ("rag", None)


def test_rag_mcp_server_exposes_structured_search_tool() -> None:
    tool_list = asyncio.run(server.list_tools())

    assert [tool.name for tool in tool_list] == ["search_knowledge"]
    result = search_knowledge("退货", top_k=2)
    assert isinstance(result, RAGSearchResult)
    assert result.query == "退货"
    assert len(result.results) <= 2
    assert all(item.content for item in result.results)


def test_rag_mcp_client_extracts_structured_result() -> None:
    result = SimpleNamespace(
        is_error=False,
        structured_content={"query": "退货", "results": [{"content": "policy"}]},
        content=[],
    )

    assert rag_mcp_client._extract_search_result(result) == {
        "query": "退货",
        "results": [{"content": "policy"}],
    }


def test_combined_route_runs_services_and_calls_deepseek_once() -> None:
    rag_service = Mock()
    rag_service.prepare.return_value = {
        "query": "帮我查订单 TEST1001 的物流状态，以及退货政策？",
        "results": [{"content": "退货需保留包装", "source": "policy.md", "chunk_index": 0}],
        "can_answer": True,
        "reliability": {"top_rerank_score": 0.9},
        "sources": [{"source": "policy.md", "index": 0}],
        "context": {"context": "退货需保留包装", "sources": [{"source": "policy.md", "index": 0}]},
        "answer": None,
    }
    mcp_service = Mock()
    mcp_service.run = AsyncMock(return_value=ORDER_DATA)
    generator = Mock()
    generator.generate_combined.return_value = "订单已发货；退货需保留包装。"
    service = ChatService(
        rag_service=rag_service,
        mcp_service=mcp_service,
        generator=generator,
    )

    query = "帮我查订单 TEST1001 的物流状态，以及退货政策？"
    result = asyncio.run(service.chat_async(query))

    assert result["answer"] == "订单已发货；退货需保留包装。"
    assert result["rag_result"] == rag_service.prepare.return_value
    assert result["mcp_result"] == ORDER_DATA
    rag_service.prepare.assert_called_once_with(query)
    mcp_service.run.assert_awaited_once_with("TEST1001")
    generator.generate_combined.assert_called_once_with(
        query,
        rag_service.prepare.return_value,
        ORDER_DATA,
    )


def test_combined_prompt_contains_both_structured_results() -> None:
    messages = build_combined_messages(
        "订单状态和退货政策？",
        {"context": "退货需保留包装"},
        ORDER_DATA,
    )

    assert "RAG 结果" in messages[1]["content"]
    assert "退货需保留包装" in messages[1]["content"]
    assert "MCP 结果" in messages[1]["content"]
    assert "TEST1001" in messages[1]["content"]


def test_deepseek_generator_has_combined_generation_method() -> None:
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="综合答案"))]
    )
    generator = DeepSeekGenerator(client=client)

    assert generator.generate_combined("问题", {"context": "规则"}, ORDER_DATA) == "综合答案"
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert "RAG 结果" in messages[1]["content"]
    assert "MCP 结果" in messages[1]["content"]
