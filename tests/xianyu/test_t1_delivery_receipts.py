"""T1 delivery receipt wiring: no receipt may create or resend a quote."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock

from fastapi.testclient import TestClient

from app.api import chat_delivery
from app.channels.xianyu.models import InboundMessage, SendReceipt
from app.channels.xianyu.stage3_worker import XianyuStage3Worker
from app.channels.xianyu.store import ChannelStore
from app.main import app


class _Sender:
    def __init__(self) -> None:
        self.calls = 0

    async def send_text(
        self, chat_id: str, buyer_id: str, text: str, request_id: str
    ) -> SendReceipt:
        del chat_id, buyer_id, text
        self.calls += 1
        return SendReceipt(request_id=request_id, local_submitted=True, platform_confirmed=True)


class _ReceiptChat:
    def __init__(self) -> None:
        self.reports: list[dict[str, str]] = []

    async def ask(self, message: object) -> dict[str, object]:
        del message
        return {
            "action": "reply",
            "answer": "当前可以 ¥1490.00包邮。",
            "turn_id": "seller:m-1",
            "proposal_id": "proposal-1",
        }

    async def report_delivery(self, **kwargs: str) -> dict[str, object]:
        self.reports.append(dict(kwargs))
        return {"accepted": True}


class _RetryReceiptChat(_ReceiptChat):
    def __init__(self) -> None:
        super().__init__()
        self.fail_once = True

    async def report_delivery(self, **kwargs: str) -> dict[str, object]:
        self.reports.append(dict(kwargs))
        if self.fail_once:
            self.fail_once = False
            raise ConnectionError("internal receipt endpoint unavailable")
        return {"accepted": True}


def test_delivery_api_requires_token_and_never_accepts_an_amount(
    monkeypatch: object,
) -> None:
    fake_service = Mock()
    fake_service.record_delivery = AsyncMock(return_value={"accepted": True})
    monkeypatch.setattr(chat_delivery, "chat_service", fake_service)
    monkeypatch.setattr(chat_delivery.settings, "xianyu_delivery_token", "test-token")
    body = {
        "chat_id": "chat-1",
        "turn_id": "seller:m-1",
        "proposal_id": "proposal-1",
        "delivery_state": "CONFIRMED",
    }

    denied = TestClient(app).post("/internal/chat/delivery", json=body)
    accepted = TestClient(app).post(
        "/internal/chat/delivery",
        json=body,
        headers={"X-Internal-Token": "test-token"},
    )
    injected_amount = TestClient(app).post(
        "/internal/chat/delivery",
        json={**body, "price_cents": 1},
        headers={"X-Internal-Token": "test-token"},
    )

    assert denied.status_code == 403
    assert accepted.status_code == 200
    assert injected_amount.status_code == 422
    fake_service.record_delivery.assert_awaited_once_with(
        "chat-1",
        turn_id="seller:m-1",
        proposal_id="proposal-1",
        delivery_state="CONFIRMED",
    )


def test_worker_reports_confirmed_delivery_without_sending_a_second_message(
    tmp_path: Path,
) -> None:
    store = ChannelStore(tmp_path / "channel.sqlite3")
    store.bind_item("seller", "listing-a", "TEST_NEG_001")
    chat = _ReceiptChat()
    worker = XianyuStage3Worker(
        store,
        account_id="seller",
        chat_client=chat,
    )
    worker.enable_account()
    message = InboundMessage(
        account_id="seller",
        platform_message_id="m-1",
        chat_id="xianyu:seller:chat-1",
        buyer_id="buyer-1",
        text="能便宜一点吗？",
        platform_item_id="listing-a",
    )

    result = asyncio.run(worker.process(message, _Sender()))
    row = store.message("seller", "m-1")

    assert result["delivery"] == "confirmed"
    assert chat.reports == [
        {
            "chat_id": "xianyu:seller:chat-1",
            "turn_id": "seller:m-1",
            "proposal_id": "proposal-1",
            "delivery_state": "CONFIRMED",
        }
    ]
    assert row and row["delivery_report_state"] == "SENT"


def test_worker_retries_a_failed_receipt_without_resending_buyer_text(tmp_path: Path) -> None:
    store = ChannelStore(tmp_path / "channel.sqlite3")
    store.bind_item("seller", "listing-a", "TEST_NEG_001")
    chat = _RetryReceiptChat()
    worker = XianyuStage3Worker(
        store,
        account_id="seller",
        chat_client=chat,
    )
    worker.enable_account()
    message = InboundMessage(
        account_id="seller",
        platform_message_id="m-1",
        chat_id="xianyu:seller:chat-1",
        buyer_id="buyer-1",
        text="能便宜一点吗？",
        platform_item_id="listing-a",
    )
    sender = _Sender()

    first = asyncio.run(worker.process(message, sender))
    replay = asyncio.run(worker.process(message, sender))

    assert first["delivery"] == "confirmed"
    assert replay["action"] == "duplicate"
    assert len(chat.reports) == 2
    assert sender.calls == 1
    assert store.message("seller", "m-1")["delivery_report_state"] == "SENT"
