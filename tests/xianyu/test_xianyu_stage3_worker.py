from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from app.api import conversations as conversations_api
from app.api import chat as chat_api
from app.channels.xianyu.action_mapper import map_chat_response
from app.channels.xianyu.models import InboundMessage, SendReceipt
from app.channels.xianyu.conversation_guard import ConversationGuard
from app.channels.xianyu.stage3_worker import HANDOFF_NOTICE, XianyuStage3Worker
from app.channels.xianyu.store import ChannelStore
from app.channels.xianyu.wecom import WeComWebhookNotifier
from app.main import app


class FakeSender:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, str]] = []

    async def send_text(self, chat_id: str, buyer_id: str, text: str, request_id: str) -> SendReceipt:
        self.calls.append((chat_id, buyer_id, text, request_id))
        return SendReceipt(request_id=request_id, local_submitted=True, platform_confirmed=True)


class FakeChat:
    def __init__(self, response: Mapping[str, Any] | BaseException) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def ask(self, *, query: str, chat_id: str, item_id: str | None) -> Mapping[str, Any]:
        self.calls.append({"query": query, "chat_id": chat_id, "item_id": item_id})
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


class BlockingChat(FakeChat):
    def __init__(self) -> None:
        super().__init__({"action": "reply", "answer": "旧答案"})
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def ask(self, *, query: str, chat_id: str, item_id: str | None) -> Mapping[str, Any]:
        self.calls.append({"query": query, "chat_id": chat_id, "item_id": item_id})
        self.started.set()
        await self.release.wait()
        return self.response


class InProcessHttpChat:
    """Exercise the real FastAPI serializer before returning to the S3 worker."""

    async def ask(
        self,
        *,
        query: str,
        chat_id: str,
        item_id: str | None,
    ) -> Mapping[str, Any]:
        response = TestClient(app).post(
            "/chat",
            json={"query": query, "chat_id": chat_id, "item_id": item_id},
        )
        response.raise_for_status()
        return response.json()


class FakeNotifier:
    def __init__(self, error: BaseException | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error = error

    async def notify_handoff(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)
        if self.error:
            raise self.error


def message(message_id: str = "m1", **changes: object) -> InboundMessage:
    data: dict[str, object] = {
        "account_id": "seller",
        "platform_message_id": message_id,
        "chat_id": "xianyu:seller:chat-1",
        "buyer_id": "buyer-1",
        "text": "这件商品多少钱？",
        "platform_item_id": "listing-a",
    }
    data.update(changes)
    return InboundMessage(**data)


def worker(tmp_path: Path, chat: FakeChat, notifier: FakeNotifier | None = None) -> tuple[XianyuStage3Worker, ChannelStore]:
    store = ChannelStore(tmp_path / "channel.sqlite3")
    store.bind_item("seller", "listing-a", "ITEM_A")
    instance = XianyuStage3Worker(
        store, account_id="seller", chat_client=chat, notifier=notifier or FakeNotifier()
    )
    instance.enable_account()
    return instance, store


def test_answer_is_mapped_sent_and_recorded_with_trusted_item(tmp_path: Path) -> None:
    chat = FakeChat({"action": "reply", "answer": "商品 A 标价 99 元。", "can_answer": True})
    instance, store = worker(tmp_path, chat)
    sender = FakeSender()

    result = asyncio.run(instance.process(message(), sender))

    assert result["action"] == "answer"
    assert sender.calls[0][:3] == ("xianyu:seller:chat-1", "buyer-1", "商品 A 标价 99 元。")
    assert chat.calls[0]["item_id"] == "ITEM_A"
    row = store.message("seller", "m1")
    assert row and row["action"] == "answer" and row["delivery_state"] == "CONFIRMED"


def test_unresolved_clarification_handoffs_with_the_fixed_buyer_reply(tmp_path: Path) -> None:
    notifier = FakeNotifier()
    chat = FakeChat({"can_answer": False, "next_step": "clarify", "answer": "请问您说的是哪一件商品？"})
    instance, store = worker(tmp_path, chat, notifier)
    sender = FakeSender()

    result = asyncio.run(instance.process(message("m2"), sender))

    assert result["action"] == "human_handoff"
    assert sender.calls[0][2] == HANDOFF_NOTICE
    assert notifier.calls[0]["reason"] == "buyer_question_requires_clarification"
    assert store.message("seller", "m2")["action"] == "human_handoff"


def test_unanswerable_question_handoffs_notifies_and_blocks_future_ai(tmp_path: Path) -> None:
    notifier = FakeNotifier()
    instance, store = worker(tmp_path, FakeChat({"action": "handoff", "reason": "refund_amount_unknown"}), notifier)
    sender = FakeSender()

    result = asyncio.run(instance.process(message("m3", text="能补偿多少？"), sender))
    later = asyncio.run(instance.process(message("m4", text="那什么时候发货？"), sender))

    assert result["action"] == "human_handoff"
    assert sender.calls[0][2] == HANDOFF_NOTICE
    assert notifier.calls[0]["reason"] == "refund_amount_unknown"
    assert store.session_state("seller", "xianyu:seller:chat-1", "buyer-1")["mode"] == "HUMAN"
    assert later == {"action": "ignored", "reason": "human_takeover"}
    assert len(sender.calls) == 1


def test_resume_auto_endpoint_restores_one_human_conversation_to_buyer_processing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat = FakeChat({"action": "handoff", "reason": "seller confirmation required"})
    instance, store = worker(tmp_path, chat)
    sender = FakeSender()
    other_chat_id = "xianyu:seller:other-chat"

    first = asyncio.run(instance.process(message("resume-handoff"), sender))
    store.set_session_mode("seller", other_chat_id, "other-buyer", "HUMAN", "other reason")
    monkeypatch.setitem(
        app.dependency_overrides,
        conversations_api.get_channel_store,
        lambda: store,
    )

    response = TestClient(app).post(
        "/conversations/xianyu:seller:chat-1/resume-auto",
        json={"account_id": "seller"},
    )

    assert first["action"] == "human_handoff"
    assert response.status_code == 200
    assert response.json()["mode"] == "AUTO"
    assert response.json()["human_takeover"] is False
    restored = store.session_state("seller", "xianyu:seller:chat-1", "buyer-1")
    assert restored["mode"] == "AUTO"
    assert restored["takeover_reason"] is None
    assert store.session_state("seller", other_chat_id, "other-buyer")["mode"] == "HUMAN"

    chat.response = {"action": "reply", "answer": "可以包邮。", "can_answer": True}
    later = asyncio.run(instance.process(message("resume-buyer", text="包邮吗？"), sender))

    assert later["action"] == "answer"
    assert chat.calls[-1]["query"] == "包邮吗？"


def test_api_error_handoffs_without_exposing_internal_error(tmp_path: Path) -> None:
    notifier = FakeNotifier(error=ConnectionError("notifier offline"))
    instance, store = worker(tmp_path, FakeChat(RuntimeError("model secret stack")), notifier)
    sender = FakeSender()

    result = asyncio.run(instance.process(message("m5"), sender))

    assert result["action"] == "human_handoff"
    row = store.message("seller", "m5")
    assert row and row["notification_state"] == "FAILED"
    assert "secret" not in sender.calls[0][2]


def test_pause_during_generation_supersedes_old_answer(tmp_path: Path) -> None:
    chat = BlockingChat()
    instance, store = worker(tmp_path, chat)
    sender = FakeSender()

    async def scenario() -> dict[str, Any]:
        task = asyncio.create_task(instance.process(message("m6"), sender))
        await chat.started.wait()
        instance.pause_account()
        chat.release.set()
        return await task

    result = asyncio.run(scenario())

    assert result["action"] == "superseded"
    assert sender.calls == []
    assert store.message("seller", "m6")["status"] == "SUPERSEDED"


def test_seller_takeover_during_generation_supersedes_old_answer(tmp_path: Path) -> None:
    chat = BlockingChat()
    instance, store = worker(tmp_path, chat)
    sender = FakeSender()

    async def scenario() -> dict[str, Any]:
        task = asyncio.create_task(instance.process(message("seller-takeover"), sender))
        await chat.started.wait()
        instance.takeover("xianyu:seller:chat-1", "buyer-1")
        chat.release.set()
        return await task

    result = asyncio.run(scenario())

    assert result == {"action": "superseded", "reason": "control_changed_during_generation"}
    assert sender.calls == []
    assert store.message("seller", "seller-takeover")["status"] == "SUPERSEDED"


def test_http_handoff_reaches_s3_once_with_reason_and_human_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reason = "缺少测光对比记录；已确认不包邮最低1470元"
    service = Mock()
    service.chat_async = AsyncMock(
        return_value={
            "query": "测光和手机对比过吗？不包邮最低多少？",
            "route": "xianyu",
            "action": "handoff",
            "answer": HANDOFF_NOTICE,
            "can_answer": False,
            "next_step": "human_handoff",
            "reason": reason,
        }
    )
    monkeypatch.setattr(chat_api, "chat_service", service)
    notifier = FakeNotifier()
    instance, store = worker(tmp_path, InProcessHttpChat(), notifier)  # type: ignore[arg-type]
    sender = FakeSender()
    inbound = message(
        "http-s3-handoff",
        text="测光和手机对比过吗？不包邮最低多少？",
    )

    first = asyncio.run(instance.process(inbound, sender))
    duplicate = asyncio.run(instance.process(inbound, sender))

    assert first["action"] == "human_handoff"
    assert duplicate == {"action": "duplicate", "message_id": "http-s3-handoff"}
    assert [call[2] for call in sender.calls] == [HANDOFF_NOTICE]
    assert len(notifier.calls) == 1
    assert notifier.calls[0]["reason"] == reason
    state = store.session_state("seller", "xianyu:seller:chat-1", "buyer-1")
    assert state["mode"] == "HUMAN"
    assert state["takeover_reason"] == reason


def test_unbound_listing_is_ignored_without_calling_chat(tmp_path: Path) -> None:
    chat = FakeChat({"action": "reply", "answer": "已确认。"})
    instance, store = worker(tmp_path, chat)
    store.bind_item("seller", "listing-b", "ITEM_B")
    sender = FakeSender()

    asyncio.run(instance.process(message("m7", platform_item_id="listing-b"), sender))
    asyncio.run(instance.process(message("m8", platform_item_id="unbound"), sender))

    assert chat.calls == [{"query": "这件商品多少钱？", "chat_id": "xianyu:seller:chat-1", "item_id": "ITEM_B"}]
    assert store.message("seller", "m8")["status"] == "IGNORED"
    assert store.message("seller", "m8")["error"] == "item_not_bound_to_seller"


def test_buyer_side_conversation_is_ignored_before_chat_send_or_handoff(tmp_path: Path) -> None:
    chat = FakeChat({"action": "handoff", "reason": "should not run"})
    notifier = FakeNotifier()
    instance, store = worker(tmp_path, chat, notifier)
    sender = FakeSender()

    results = [
        asyncio.run(
            instance.process(
                message(f"buyer-side-{index}", platform_item_id="another-sellers-listing"),
                sender,
            )
        )
        for index in range(10)
    ]

    assert results == [{"action": "ignored", "reason": "item_not_bound_to_seller"}] * 10
    assert chat.calls == []
    assert sender.calls == []
    assert notifier.calls == []
    assert store.message("seller", "buyer-side-0")["status"] == "IGNORED"


def test_paused_ai_is_ignored_before_chat(tmp_path: Path) -> None:
    chat = FakeChat({"action": "reply", "answer": "should not run"})
    instance, _ = worker(tmp_path, chat)
    sender = FakeSender()
    instance.pause_account()

    result = asyncio.run(instance.process(message("paused"), sender))

    assert result == {"action": "ignored", "reason": "ai_disabled"}
    assert chat.calls == []
    assert sender.calls == []


def test_missing_listing_id_is_ignored_without_prior_bound_conversation(tmp_path: Path) -> None:
    chat = FakeChat({"action": "reply", "answer": "已确认。"})
    instance, _ = worker(tmp_path, chat)
    sender = FakeSender()

    result = asyncio.run(
        instance.process(message("no-listing", chat_id="xianyu:seller:new-chat", platform_item_id=None), sender)
    )

    assert result == {"action": "ignored", "reason": "missing_platform_item_id"}
    assert chat.calls == []
    assert sender.calls == []


def test_followups_reuse_the_verified_channel_item_context(tmp_path: Path) -> None:
    chat = FakeChat({"action": "reply", "answer": "已确认。"})
    instance, store = worker(tmp_path, chat)
    sender = FakeSender()

    first = asyncio.run(instance.process(message("first", text="这个相机还在吗？"), sender))
    second = asyncio.run(
        instance.process(message("second", text="这是 50mm 镜头吗？", platform_item_id=None), sender)
    )
    third = asyncio.run(
        instance.process(message("third", text="这个镜头一起出吗？", platform_item_id=None), sender)
    )

    assert [first["action"], second["action"], third["action"]] == ["answer", "answer", "answer"]
    assert [call["item_id"] for call in chat.calls] == ["ITEM_A", "ITEM_A", "ITEM_A"]
    session = store.session_state("seller", "xianyu:seller:chat-1", "buyer-1")
    assert session["current_item_id"] == "ITEM_A"
    assert session["updated_at"]


def test_new_bound_listing_replaces_the_channel_item_context(tmp_path: Path) -> None:
    chat = FakeChat({"action": "reply", "answer": "已确认。"})
    instance, store = worker(tmp_path, chat)
    store.bind_item("seller", "listing-b", "ITEM_B")
    sender = FakeSender()

    asyncio.run(instance.process(message("first", platform_item_id="listing-a"), sender))
    asyncio.run(instance.process(message("second", platform_item_id="listing-b"), sender))
    followup = asyncio.run(instance.process(message("third", platform_item_id=None), sender))

    assert followup["action"] == "answer"
    assert [call["item_id"] for call in chat.calls] == ["ITEM_A", "ITEM_B", "ITEM_B"]
    assert store.session_state("seller", "xianyu:seller:chat-1", "buyer-1")["current_item_id"] == "ITEM_B"


def test_conversation_guard_diagnostic_is_hashed_and_explains_allow(tmp_path: Path) -> None:
    store = ChannelStore(tmp_path / "channel.sqlite3")
    store.bind_item("seller", "listing-a", "ITEM_A")
    store.set_enabled("seller", True)
    diagnostics: list[Mapping[str, object]] = []
    guard = ConversationGuard(
        store,
        account_id="seller",
        diagnostic_sink=diagnostics.append,
    )

    decision = guard.check(message("diagnostic-message"))

    assert decision.action == "ALLOW"
    assert len(diagnostics) == 1
    record = diagnostics[0]
    assert record["decision"] == "ALLOW"
    assert record["reason"] == "seller_buyer_bound_item"
    assert record["sender_id_equals_account"] is False
    assert record["raw_item_id_hash"] != "listing-a"
    assert record["mapped_item_id"] == "ITEM_A"
    assert record["is_item_bound"] is True
    assert record["item_source"] == "event"
    assert "allow_reason" in record
    assert "listing-a" not in str(record)
    assert "diagnostic-message" not in str(record)
    assert "xianyu:seller:chat-1" not in str(record)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"action": "reply", "answer": "ok"}, "answer"),
        ({"action": "clarify", "answer": "which?"}, "human_handoff"),
        ({"action": "reply", "answer": ""}, "human_handoff"),
        ({"action": "unknown", "answer": "unsafe"}, "human_handoff"),
    ],
)
def test_action_mapper_is_fail_closed(payload: Mapping[str, Any], expected: str) -> None:
    assert map_chat_response(payload).action == expected


def test_wecom_webhook_rejects_noncanonical_or_insecure_urls() -> None:
    with pytest.raises(ValueError):
        WeComWebhookNotifier("http://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=x")
    with pytest.raises(ValueError):
        WeComWebhookNotifier("https://example.test/hook?key=x")
    with pytest.raises(ValueError):
        WeComWebhookNotifier("https://qyapi.weixin.qq.com/other?key=x")
