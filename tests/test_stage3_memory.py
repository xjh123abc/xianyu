"""Acceptance tests for Stage 3 short-term conversation memory."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

from fastapi.testclient import TestClient

from app.api import chat as chat_api
from app.main import app
from app.services.chat_service import ChatService, route_query
from app.services.session_manager import SessionManager


ORDER_RESULT = {
    "found": True,
    "order_id": "TEST1001",
    "order_status": "待发货",
    "logistics_status": "暂无物流",
    "tracking_no": None,
}


def test_context_aware_router_handles_the_three_documented_turns() -> None:
    state = {"order_id": "TEST1001", "item_id": None, "last_intent": "order_query"}
    history = [
        {"role": "user", "content": "帮我查 TEST1001"},
        {"role": "assistant", "content": "TEST1001 当前待发货"},
    ]

    assert route_query("帮我查 TEST1001") == ("order", "TEST1001")
    assert route_query("那它一般多久发货？", history, state) == ("rag", None)
    assert route_query("那它现在是不是已经超时了？", history, state) == (
        "rag_mcp",
        "TEST1001",
    )


def test_session_manager_isolates_chat_ids_and_limits_history() -> None:
    manager = SessionManager(max_history=4)
    manager.append_turn(
        "chat_001",
        "查 TEST1001",
        "已找到",
        order_id="TEST1001",
        item_id="DEMO_ITEM_001",
        last_intent="order_query",
    )
    manager.append_turn("chat_001", "那多久发货？", "通常 24 小时", last_intent="rag_query")
    manager.append_turn("chat_002", "查 TEST1002", "已找到", order_id="TEST1002", last_intent="order_query")

    _, first = manager.get_or_create("chat_001")
    _, second = manager.get_or_create("chat_002")
    assert first["state"]["order_id"] == "TEST1001"
    assert first["state"]["item_id"] == "DEMO_ITEM_001"
    assert first["state"]["last_intent"] == "rag_query"
    assert len(first["history"]) == 4
    assert second["state"]["order_id"] == "TEST1002"
    assert second["state"]["item_id"] is None
    assert second["history"] != first["history"]


def test_chat_service_preserves_order_context_across_three_turns() -> None:
    query_one = "帮我查 TEST1001"
    query_two = "那它一般多久发货？"
    query_three = "那它现在是不是已经超时了？"

    rag_service = Mock()
    rag_service.chat.return_value = {"query": query_two, "answer": "通常 24 小时内发货", "results": []}
    rag_service.prepare.return_value = {
        "query": query_three,
        "results": [{"content": "付款成功后 24 小时内发出。"}],
        "context": {"context": "付款成功后 24 小时内发出。", "sources": [{"source": "shipping.md", "index": 0}]},
        "sources": [{"source": "shipping.md", "index": 0}],
        "reliability": {"can_answer": True},
    }
    mcp_service = Mock()
    mcp_service.get_order = AsyncMock(return_value=ORDER_RESULT)
    generator = Mock()
    generator.generate_order.return_value = "TEST1001 当前待发货"
    generator.generate_combined.return_value = "TEST1001 当前待发货，平台规则为 24 小时内发货。"
    service = ChatService(
        rag_service=rag_service,
        mcp_service=mcp_service,
        generator=generator,
        session_manager=SessionManager(),
    )

    first = asyncio.run(service.chat_async(query_one, "chat_001"))
    second = asyncio.run(service.chat_async(query_two, "chat_001"))
    third = asyncio.run(service.chat_async(query_three, "chat_001"))

    assert first["chat_id"] == second["chat_id"] == third["chat_id"] == "chat_001"
    assert second["answer"] == "通常 24 小时内发货"
    assert third["answer"].startswith("TEST1001 当前待发货")
    assert mcp_service.get_order.await_count == 2
    generator.generate_combined.assert_called_once()
    assert generator.generate_combined.call_args.kwargs["history"]


def test_chat_api_accepts_and_returns_chat_id(monkeypatch) -> None:
    rag_service = Mock()
    rag_service.chat.return_value = {"query": "一般多久发货？", "results": []}
    monkeypatch.setattr(
        chat_api,
        "chat_service",
        ChatService(rag_service=rag_service, session_manager=SessionManager()),
    )

    response = TestClient(app).post(
        "/chat",
        json={"chat_id": "chat_api_001", "query": "一般多久发货？"},
    )

    assert response.status_code == 200
    assert response.json()["chat_id"] == "chat_api_001"
    rag_service.chat.assert_called_once_with("一般多久发货？")
