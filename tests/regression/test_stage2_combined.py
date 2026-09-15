"""Acceptance tests for the Stage 2 RAG + MCP workflow."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

from app.services.chat_service import ChatService, route_query


ORDER_RESULT = {
    "found": True,
    "order_id": "TEST1001",
    "order_status": "已发货",
    "logistics_status": "运输中",
    "tracking_no": "MOCKEXP1001",
}


def test_v1_examples_select_rag_mcp_only_for_combined_question() -> None:
    assert route_query("TEST1001 现在是什么状态？一般多久发货？") == (
        "rag_mcp",
        "TEST1001",
    )
    assert route_query("一般多久发货？") == ("rag", None)
    assert route_query("TEST1001 现在是什么状态？") == ("order", "TEST1001")


def test_rag_mcp_executes_both_services_before_final_generation() -> None:
    query = "TEST1001 现在是什么状态？一般多久发货？"
    rag_result = {
        "query": query,
        "results": [{"content": "付款成功后 24 小时内发出。"}],
        "context": {"context": "付款成功后 24 小时内发出。", "sources": [{"source": "shipping.md", "index": 0}]},
        "sources": [{"source": "shipping.md", "index": 0}],
        "reliability": {"can_answer": True},
    }
    rag_service = Mock()
    rag_service.prepare.return_value = rag_result
    mcp_service = Mock()
    mcp_service.get_order = AsyncMock(return_value=ORDER_RESULT)
    generator = Mock()
    generator.generate_combined.return_value = (
        "订单 TEST1001 当前已发货，平台规则为付款成功后 24 小时内发出。"
    )

    result = asyncio.run(
        ChatService(
            rag_service=rag_service,
            mcp_service=mcp_service,
            generator=generator,
        ).chat_async(query)
    )

    assert result["route"] == "rag_mcp"
    assert result["answer"].startswith("订单 TEST1001")
    rag_service.prepare.assert_called_once_with(query)
    mcp_service.get_order.assert_awaited_once_with("TEST1001")
    generator.generate_combined.assert_called_once_with(
        query,
        rag_result,
        ORDER_RESULT,
    )
