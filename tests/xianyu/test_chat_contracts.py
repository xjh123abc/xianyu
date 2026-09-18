from __future__ import annotations

import asyncio

import pytest

from app.channels.xianyu.adapter import to_chat_message
from app.channels.xianyu.chat_client import ChatApiClient
from app.channels.xianyu.models import InboundMessage
from app.services.chat_contracts import (
    ChatMessage,
    ChatResponse,
    SessionContext,
    Task,
    TaskResult,
    default_negotiation_state,
)


def _inbound(**changes: object) -> InboundMessage:
    values: dict[str, object] = {
        "account_id": "seller-1",
        "platform_message_id": "event-1",
        "chat_id": "xianyu:seller-1:chat-1",
        "buyer_id": "buyer-1",
        "text": "  这个相机带镜头吗？  ",
        "platform_item_id": "untrusted-listing-id",
    }
    values.update(changes)
    return InboundMessage(**values)


def test_xianyu_adapter_creates_the_transport_independent_chat_message() -> None:
    message = to_chat_message(_inbound(), item_id="ITEM_001")

    assert message.platform == "xianyu"
    assert message.account_id == "seller-1"
    assert message.chat_id == "xianyu:seller-1:chat-1"
    assert message.buyer_id == "buyer-1"
    assert message.item_id == "ITEM_001"
    assert message.text == "这个相机带镜头吗？"


def test_chat_api_client_keeps_the_existing_chat_wire_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[tuple[str, dict[str, object]]] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"action": "reply", "answer": "已确认"}

    class FakeAsyncClient:
        def __init__(self, *, timeout: float) -> None:
            assert timeout == 12.0

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def post(self, url: str, *, json: dict[str, object]) -> FakeResponse:
            sent.append((url, json))
            return FakeResponse()

    monkeypatch.setattr("app.channels.xianyu.chat_client.httpx.AsyncClient", FakeAsyncClient)
    response = asyncio.run(
        ChatApiClient("http://chat.test/", 12.0).ask(
            ChatMessage("xianyu", "seller-1", "chat-1", "buyer-1", "ITEM_001", "你好")
        )
    )

    assert response == {"action": "reply", "answer": "已确认"}
    assert sent == [("http://chat.test/chat", {"query": "你好", "chat_id": "chat-1", "item_id": "ITEM_001"})]


def test_session_contract_reserves_independent_negotiation_state() -> None:
    first = SessionContext()
    second = SessionContext()
    first.negotiation["round"] = 2

    assert second.negotiation == default_negotiation_state()
    assert first.negotiation["item_id"] is None


def test_task_and_response_contracts_preserve_partial_task_outcomes() -> None:
    product = Task("product-1", "product", "这个修过吗？")
    unavailable_service = TaskResult(
        task_id="service-1",
        status="unavailable",
        answer="",
        reason="shipping_record_missing",
    )
    response = ChatResponse(
        action="reply",
        answer="没有维修记录。周日是否发货暂时无法确认。",
        results=[TaskResult(product.task_id, "answered", "没有维修记录。"), unavailable_service],
    )

    assert [result.status for result in response.results] == ["answered", "unavailable"]
    assert response.results[1].reason == "shipping_record_missing"


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: Task("", "product", "问题"), "task_id"),
        (lambda: Task("task", "invalid", "问题"), "task_type"),
        (lambda: SessionContext(last_task_type="invalid"), "last_task_type"),
        (lambda: SessionContext(negotiation={"round": -1}), "negotiation.round"),
    ],
)
def test_contracts_reject_invalid_planning_state(factory: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()
