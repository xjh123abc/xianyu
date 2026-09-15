"""Acceptance tests for Stage 3 short-term conversation memory."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
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
    state = {
        "order_id": "TEST1001",
        "current_item_id": None,
        "last_intent": "order_query",
    }
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
        last_intent="order_query",
    )
    manager.set_current_item_id("chat_001", "DEMO_ITEM_001")
    manager.append_turn("chat_001", "那多久发货？", "通常 24 小时", last_intent="rag_query")
    manager.append_turn("chat_002", "查 TEST1002", "已找到", order_id="TEST1002", last_intent="order_query")

    _, first = manager.get_or_create("chat_001")
    _, second = manager.get_or_create("chat_002")
    assert first["state"]["order_id"] == "TEST1001"
    assert first["state"]["current_item_id"] == "DEMO_ITEM_001"
    assert first["state"]["last_intent"] == "rag_query"
    assert len(first["history"]) == 4
    assert second["state"]["order_id"] == "TEST1002"
    assert second["state"]["current_item_id"] is None
    assert second["history"] != first["history"]


def test_session_manager_expires_idle_sessions_and_limits_total_count() -> None:
    now = [100.0]
    manager = SessionManager(
        ttl_seconds=10,
        max_sessions=2,
        clock=lambda: now[0],
    )
    manager.get_or_create("old")
    now[0] = 101.0
    manager.get_or_create("middle")
    now[0] = 102.0
    manager.get_or_create("new")

    assert set(manager.sessions) == {"middle", "new"}

    now[0] = 112.0
    _, expired = manager.get_or_create("middle")
    assert expired["history"] == []
    assert expired["state"]["order_id"] is None


@pytest.fixture
def session_database_path():
    database_path = Path("tests/.session-manager-test.sqlite3").resolve()
    related_paths = [
        database_path,
        Path(f"{database_path}-wal"),
        Path(f"{database_path}-shm"),
    ]
    for path in related_paths:
        path.unlink(missing_ok=True)
    try:
        yield database_path
    finally:
        for path in related_paths:
            path.unlink(missing_ok=True)


def test_sqlite_session_state_is_shared_between_manager_instances(
    session_database_path,
) -> None:
    database_path = session_database_path
    first = SessionManager(database_path=database_path)
    second = SessionManager(database_path=database_path)
    first.append_turn(
        "shared",
        "query",
        "answer",
        order_id="TEST1001",
    )

    _, restored = second.get_or_create("shared")

    assert restored["state"]["order_id"] == "TEST1001"
    assert restored["history"][-1] == {"role": "assistant", "content": "answer"}


def test_sqlite_sessions_enforce_ttl_and_capacity(session_database_path) -> None:
    now = [100.0]
    manager = SessionManager(
        database_path=session_database_path,
        ttl_seconds=10,
        max_sessions=2,
        clock=lambda: now[0],
    )
    manager.append_turn("oldest", "q", "a", order_id="TEST1001")
    now[0] = 101.0
    manager.append_turn("middle", "q", "a", order_id="TEST1002")
    now[0] = 102.0
    manager.append_turn("newest", "q", "a", order_id="TEST1003")

    _, evicted = manager.get_or_create("oldest")
    assert evicted["state"]["order_id"] is None

    now[0] = 113.0
    _, expired = manager.get_or_create("newest")
    assert expired["state"]["order_id"] is None


def test_sqlite_session_lock_serializes_workers(session_database_path) -> None:
    database_path = session_database_path
    first = SessionManager(database_path=database_path, lock_timeout_seconds=1)
    second = SessionManager(database_path=database_path, lock_timeout_seconds=1)
    events = []

    async def worker(manager, name, delay):
        async with manager.session_lock("shared"):
            events.append(f"{name}-start")
            await asyncio.sleep(delay)
            events.append(f"{name}-end")

    async def exercise():
        first_task = asyncio.create_task(worker(first, "first", 0.05))
        await asyncio.sleep(0.01)
        second_task = asyncio.create_task(worker(second, "second", 0))
        await asyncio.gather(first_task, second_task)

    asyncio.run(exercise())

    assert events == ["first-start", "first-end", "second-start", "second-end"]


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
