"""HTTP contract checks for the S5/S6 Xianyu expert decision."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

from fastapi.testclient import TestClient

from app.api import chat as chat_api
from app.main import app


def test_unavailable_reason_survives_full_http_serialization(monkeypatch) -> None:
    service = Mock()
    service.chat_async = AsyncMock(
        return_value={
            "query": "测光和手机对比过吗？不包邮最低多少？",
            "chat_id": "s6_http_handoff",
            "item_id": "CANON_FTB_001",
            "route": "xianyu",
            "action": "reply",
            "answer": "该问题目前暂无足够信息确认。",
            "can_answer": False,
            "next_step": None,
            "reason": "缺少测光对比记录；已确认不包邮最低1470元",
        }
    )
    monkeypatch.setattr(chat_api, "chat_service", service)

    response = TestClient(app).post(
        "/chat",
        json={
            "query": "测光和手机对比过吗？不包邮最低多少？",
            "chat_id": "s6_http_handoff",
            "item_id": "CANON_FTB_001",
        },
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "该问题目前暂无足够信息确认。"
    assert response.json()["reason"] == "缺少测光对比记录；已确认不包邮最低1470元"
    service.chat_async.assert_awaited_once()


def test_http_schema_exposes_reason_without_exposing_internal_tasks() -> None:
    schema = app.openapi()["components"]["schemas"]["Response"]["properties"]

    assert "reason" in schema
    assert "tasks" not in schema


def test_http_schema_exposes_expert_debug_trace(monkeypatch) -> None:
    service = Mock()
    service.chat_async = AsyncMock(
        return_value={
            "query": "售后怎么处理？",
            "chat_id": "safety_trace",
            "route": "xianyu",
            "action": "reply",
            "answer": "模型原始回答",
            "raw_answer": "模型原始回答",
            "evidence": "实际 rerank 证据",
            "can_answer": True,
        }
    )
    monkeypatch.setattr(chat_api, "chat_service", service)

    response = TestClient(app).post(
        "/chat", json={"query": "售后怎么处理？", "chat_id": "safety_trace"}
    )

    assert response.status_code == 200
    assert response.json()["evidence"] == "实际 rerank 证据"
    assert response.json()["raw_answer"] == response.json()["answer"] == "模型原始回答"
