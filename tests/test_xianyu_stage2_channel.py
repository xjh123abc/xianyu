from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

import pytest

from app.channels.xianyu.adapter import iter_sync_events
from app.channels.xianyu.client import WebSocketTextSender, build_text_payload
from app.channels.xianyu.lock import AccountProcessLock
from app.channels.xianyu.models import InboundMessage, SendReceipt
from app.channels.xianyu.store import ChannelStore
from app.channels.xianyu.worker import XianyuStage2Worker


class FakeSender:
    def __init__(self, result: SendReceipt | None = None, error: BaseException | None = None) -> None:
        self.calls: list[tuple[str, str, str, str]] = []
        self.result = result
        self.error = error

    async def send_text(self, chat_id: str, buyer_id: str, text: str, request_id: str) -> SendReceipt:
        self.calls.append((chat_id, buyer_id, text, request_id))
        if self.error:
            raise self.error
        return self.result or SendReceipt(request_id=request_id, local_submitted=True)


def _message(message_id: str = "m1", **kwargs: object) -> InboundMessage:
    values: dict[str, object] = {
        "account_id": "seller",
        "platform_message_id": message_id,
        "chat_id": "chat-1",
        "buyer_id": "buyer-1",
        "text": "闲鱼渠道接通测试",
    }
    values.update(kwargs)
    return InboundMessage(**values)


def _worker(tmp_path: Path) -> tuple[XianyuStage2Worker, ChannelStore]:
    store = ChannelStore(tmp_path / "channel.sqlite3")
    worker = XianyuStage2Worker(
        store,
        account_id="seller",
        trigger_text="闲鱼渠道接通测试",
    )
    worker.enable_account()
    return worker, store


def test_worker_deduplicates_and_records_confirmed_delivery(tmp_path: Path) -> None:
    worker, store = _worker(tmp_path)
    sender = FakeSender(SendReceipt(request_id="ignored", local_submitted=True, platform_confirmed=True))

    first = asyncio.run(worker.process(_message(), sender))
    duplicate_results = [asyncio.run(worker.process(_message(), sender)) for _ in range(3)]

    assert first["action"] == "confirmed"
    assert all(result["action"] == "duplicate" for result in duplicate_results)
    assert len(sender.calls) == 1
    row = store.message("seller", "m1")
    assert row and row["status"] == "CONFIRMED" and row["delivery_state"] == "CONFIRMED"


@pytest.mark.parametrize("control", ["pause", "takeover"])
def test_pause_or_takeover_blocks_new_send(tmp_path: Path, control: str) -> None:
    worker, _ = _worker(tmp_path)
    if control == "pause":
        worker.pause_account()
    else:
        worker.takeover("chat-1", "buyer-1")
    sender = FakeSender()

    result = asyncio.run(worker.process(_message("m2"), sender))

    assert result["action"] == "blocked"
    assert sender.calls == []


def test_seller_echo_and_non_text_are_ignored(tmp_path: Path) -> None:
    worker, store = _worker(tmp_path)
    sender = FakeSender()

    seller = asyncio.run(worker.process(_message("m3", sender_is_seller=True), sender))
    attachment = asyncio.run(worker.process(_message("m4", message_type="attachment"), sender))

    assert seller["reason"] == "seller_echo"
    assert attachment["reason"] == "non_text_or_system"
    assert sender.calls == []
    assert store.message("seller", "m3")["status"] == "IGNORED"


@pytest.mark.parametrize(
    ("error", "expected"),
    [(ConnectionError("offline"), "failed"), (TimeoutError("uncertain"), "unknown")],
)
def test_send_failures_are_distinguished(tmp_path: Path, error: BaseException, expected: str) -> None:
    worker, store = _worker(tmp_path)
    result = asyncio.run(worker.process(_message("m5"), FakeSender(error=error)))

    assert result["action"] == expected
    assert store.message("seller", "m5")["delivery_state"] == expected.upper()


def test_account_lock_rejects_second_owner_and_releases(tmp_path: Path) -> None:
    lock_path = tmp_path / "seller.lock"
    first = AccountProcessLock(lock_path)
    second = AccountProcessLock(lock_path)
    first.acquire()
    try:
        with pytest.raises(RuntimeError):
            second.acquire()
    finally:
        first.release()
    second.acquire()
    second.release()


def test_payload_contains_only_validated_receiver_and_fixed_text() -> None:
    payload = build_text_payload(
        chat_id="chat-1",
        buyer_id="buyer-1",
        seller_id="seller",
        text="闲鱼AI客服接通测试成功",
        request_id="request-1",
    )
    assert payload["body"][0]["cid"] == "chat-1@goofish"
    assert payload["body"][1]["actualReceivers"] == ["buyer-1@goofish", "seller@goofish"]
    encoded = payload["body"][0]["content"]["custom"]["data"]
    decoded = json.loads(base64.b64decode(encoded).decode("utf-8"))
    assert decoded["text"]["text"] == "闲鱼AI客服接通测试成功"


def test_adapter_processes_every_record_and_marks_seller() -> None:
    def encoded(event: dict[str, object]) -> str:
        return base64.b64encode(json.dumps(event).encode()).decode()

    events = [
        {"1": {"2": "chat-1@goofish", "5": 1, "10": {"senderUserId": "buyer", "reminderContent": "hi"}}},
        {"1": {"2": "chat-1@goofish", "5": 2, "10": {"senderUserId": "seller", "reminderContent": "ok"}}},
    ]
    message = {"body": {"syncPushPackage": {"data": [{"id": "m1", "data": encoded(events[0])}, {"id": "m2", "data": encoded(events[1])}]}}}

    parsed = list(iter_sync_events(message, account_id="seller", seller_id="seller", decrypt=lambda _: ""))

    assert [item.platform_message_id for item in parsed] == ["m1", "m2"]
    assert parsed[0].chat_id == "xianyu:seller:chat-1"
    assert parsed[0].platform_chat_id == "chat-1"
    assert parsed[0].sender_is_seller is False
    assert parsed[1].sender_is_seller is True


def test_sender_uses_injected_platform_uuid_format() -> None:
    captured: list[dict[str, object]] = []

    async def capture(payload: dict[str, object]) -> None:
        captured.append(payload)

    sender = WebSocketTextSender(
        websocket=None,
        seller_id="seller",
        send_json=capture,
        uuid_factory=lambda: "-12345678901",
        mid_factory=lambda: "1234567890 0",
    )

    receipt = asyncio.run(sender.send_text("chat-1", "buyer-1", "hello", "request-1"))

    assert receipt.local_submitted is True
    assert captured[0]["body"][0]["uuid"] == "-12345678901"
    assert captured[0]["headers"]["mid"] == "1234567890 0"
