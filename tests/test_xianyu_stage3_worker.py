from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from app.channels.xianyu.action_mapper import map_chat_response
from app.channels.xianyu.models import InboundMessage, SendReceipt
from app.channels.xianyu.stage3_worker import HANDOFF_NOTICE, XianyuStage3Worker
from app.channels.xianyu.store import ChannelStore
from app.channels.xianyu.wecom import WeComWebhookNotifier


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


def test_clarification_is_sent_even_when_api_cannot_answer(tmp_path: Path) -> None:
    chat = FakeChat({"can_answer": False, "next_step": "clarify", "answer": "请问您说的是哪一件商品？"})
    instance, store = worker(tmp_path, chat)
    sender = FakeSender()

    result = asyncio.run(instance.process(message("m2"), sender))

    assert result["action"] == "clarify"
    assert sender.calls[0][2] == "请问您说的是哪一件商品？"
    assert store.message("seller", "m2")["action"] == "clarify"


def test_unanswerable_question_handoffs_notifies_and_blocks_future_ai(tmp_path: Path) -> None:
    notifier = FakeNotifier()
    instance, store = worker(tmp_path, FakeChat({"action": "handoff", "reason": "退款金额需要卖家确认"}), notifier)
    sender = FakeSender()

    result = asyncio.run(instance.process(message("m3", text="能补偿多少？"), sender))
    later = asyncio.run(instance.process(message("m4", text="那什么时候发货？"), sender))

    assert result["action"] == "human_handoff"
    assert sender.calls[0][2] == HANDOFF_NOTICE
    assert notifier.calls[0]["reason"] == "退款金额需要卖家确认"
    assert store.session_state("seller", "xianyu:seller:chat-1", "buyer-1")["mode"] == "HUMAN"
    assert later["action"] == "blocked"
    assert len(sender.calls) == 1


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


def test_bound_listing_switches_current_item_but_unbound_never_reuses_old_item(tmp_path: Path) -> None:
    chat = FakeChat({"action": "reply", "answer": "已确认。"})
    instance, store = worker(tmp_path, chat)
    store.bind_item("seller", "listing-b", "ITEM_B")
    sender = FakeSender()

    asyncio.run(instance.process(message("m7", platform_item_id="listing-b"), sender))
    asyncio.run(instance.process(message("m8", platform_item_id="unbound"), sender))

    assert chat.calls[0]["item_id"] == "ITEM_B"
    assert chat.calls[1]["item_id"] is None


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"action": "reply", "answer": "ok"}, "answer"),
        ({"action": "clarify", "answer": "which?"}, "clarify"),
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
