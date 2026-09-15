"""Service and API tests for the Step 5 order branch."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from app.api import chat as chat_api
from app.generation.deepseek import DeepSeekGenerator
from app.generation.prompt import build_order_messages
from app.main import app
from app.services import chat_service as chat_service_module
from app.services.chat_service import ChatService


ORDER_DATA = {
    "found": True,
    "order_id": "TEST1003",
    "order_status": "已签收",
    "logistics_status": "已签收",
    "tracking_no": "MOCKEXP1003",
}


class FakeOrderGenerator:
    def __init__(self, answer: str = "根据本地模拟订单数据，订单 TEST1003 已签收。") -> None:
        self.answer = answer
        self.calls: list[tuple[str, dict[str, object]]] = []

    def generate_order(self, query: str, order_data: dict[str, object]) -> str:
        self.calls.append((query, order_data))
        return self.answer


class FailingOrderGenerator(FakeOrderGenerator):
    def generate_order(self, query: str, order_data: dict[str, object]) -> str:
        raise RuntimeError("generation failed")


def test_order_prompt_uses_structured_json_and_order_rules() -> None:
    messages = build_order_messages("帮我查订单 TEST1003", ORDER_DATA)

    assert "本地模拟订单数据" in messages[0]["content"]
    assert "运单号为空时说明暂无运单号" in messages[0]["content"]
    assert "用户问题：\n帮我查订单 TEST1003" in messages[1]["content"]
    json_text = messages[1]["content"].split("订单数据：\n", 1)[1]
    assert json.loads(json_text) == ORDER_DATA


def test_deepseek_order_generation_reuses_existing_completion_client() -> None:
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="订单已签收。"))]
    )
    generator = DeepSeekGenerator(client=client)

    answer = generator.generate_order("帮我查订单 TEST1003", ORDER_DATA)

    assert answer == "订单已签收。"
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["messages"][1]["content"].endswith(
        json.dumps(ORDER_DATA, ensure_ascii=False)
    )


def test_order_branch_calls_mcp_and_deepseek(monkeypatch: pytest.MonkeyPatch) -> None:
    generator = FakeOrderGenerator()
    mcp_lookup = AsyncMock(return_value=ORDER_DATA)
    monkeypatch.setattr(chat_service_module, "get_order_via_mcp", mcp_lookup)
    service = ChatService(generator=generator)

    result = asyncio.run(
        service.chat_async("TEST1003帮我查询一下这个订单号")
    )

    assert result == {
        "query": "TEST1003帮我查询一下这个订单号",
        "answer": "根据本地模拟订单数据，订单 TEST1003 已签收。",
        "sources": [],
        "results": [],
        "reliability": None,
        "next_step": None,
        "can_answer": True,
    }
    mcp_lookup.assert_awaited_once_with("TEST1003")
    assert generator.calls == [
        ("TEST1003帮我查询一下这个订单号", ORDER_DATA)
    ]


def test_rag_branch_keeps_existing_chat_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mcp_lookup = AsyncMock()
    monkeypatch.setattr(chat_service_module, "get_order_via_mcp", mcp_lookup)
    service = ChatService(generator=FakeOrderGenerator())
    existing_rag_chat = Mock(
        return_value={"query": "订单一般多久发货？", "results": []}
    )
    monkeypatch.setattr(service, "chat", existing_rag_chat)

    result = asyncio.run(service.chat_async("订单一般多久发货？"))

    assert result == {"query": "订单一般多久发货？", "results": []}
    existing_rag_chat.assert_called_once_with("订单一般多久发货？")
    mcp_lookup.assert_not_awaited()


@pytest.mark.parametrize(
    ("query", "expected_answer"),
    [
        (
            "帮我查一下我的订单。",
            "请提供订单号，并重新发送完整问题，例如：帮我查订单 TEST1001。",
        ),
        (
            "帮我取消订单 TEST1001。",
            "本版本暂不支持取消订单、退款或修改地址等操作，仅支持订单状态查询。",
        ),
    ],
)
def test_non_order_branches_do_not_call_mcp_or_deepseek(
    monkeypatch: pytest.MonkeyPatch,
    query: str,
    expected_answer: str,
) -> None:
    mcp_lookup = AsyncMock()
    monkeypatch.setattr(chat_service_module, "get_order_via_mcp", mcp_lookup)
    generator = FakeOrderGenerator()
    service = ChatService(generator=generator)

    result = asyncio.run(service.chat_async(query))

    assert result["answer"] == expected_answer
    assert result["can_answer"] is False
    assert result["sources"] == []
    assert result["results"] == []
    assert result["reliability"] is None
    assert result["next_step"] is None
    mcp_lookup.assert_not_awaited()
    assert generator.calls == []


def test_not_found_order_is_successful_query_without_model_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mcp_lookup = AsyncMock(
        return_value={
            "found": False,
            "order_id": "TEST9999",
            "order_status": None,
            "logistics_status": None,
            "tracking_no": None,
        }
    )
    monkeypatch.setattr(chat_service_module, "get_order_via_mcp", mcp_lookup)
    generator = FakeOrderGenerator()
    service = ChatService(generator=generator)

    result = asyncio.run(service.chat_async("帮我查订单 TEST9999"))

    assert result["answer"] == "未查询到模拟订单 TEST9999，请核对订单号。"
    assert result["can_answer"] is True
    assert result["reliability"] is None
    mcp_lookup.assert_awaited_once_with("TEST9999")
    assert generator.calls == []


def test_mcp_failure_is_reported_without_success_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mcp_lookup = AsyncMock(side_effect=RuntimeError("server unavailable"))
    monkeypatch.setattr(chat_service_module, "get_order_via_mcp", mcp_lookup)
    service = ChatService(generator=FakeOrderGenerator())

    result = asyncio.run(service.chat_async("帮我查订单 TEST1001"))

    assert result["answer"] == "本次订单查询失败，请稍后重试。"
    assert result["can_answer"] is False
    assert result["reliability"] is None
    assert result["next_step"] is None


def test_order_generation_failure_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mcp_lookup = AsyncMock(return_value=ORDER_DATA)
    monkeypatch.setattr(chat_service_module, "get_order_via_mcp", mcp_lookup)
    service = ChatService(generator=FailingOrderGenerator())

    result = asyncio.run(service.chat_async("帮我查订单 TEST1003"))

    assert result["answer"] == "本次订单查询回答失败，请稍后重试。"
    assert result["can_answer"] is False
    assert result["sources"] == []


def test_chat_api_serializes_order_branch_with_minimal_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mcp_lookup = AsyncMock(return_value=ORDER_DATA)
    monkeypatch.setattr(chat_service_module, "get_order_via_mcp", mcp_lookup)
    monkeypatch.setattr(
        chat_api,
        "chat_service",
        ChatService(generator=FakeOrderGenerator()),
    )

    response = TestClient(app).post(
        "/chat",
        json={"query": "帮我查订单 TEST1003", "chat_id": "chat_order_001"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "query": "帮我查订单 TEST1003",
        "chat_id": "chat_order_001",
        "answer": "根据本地模拟订单数据，订单 TEST1003 已签收。",
        "sources": [],
        "can_answer": True,
        "next_step": None,
        "reliability": None,
        "results": [],
    }
