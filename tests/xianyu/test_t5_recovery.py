from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.channels.xianyu.models import InboundMessage
from app.channels.xianyu.store import ChannelStore
from app.services import chat_service as chat_service_module
from app.services.chat_service import ChatService
from app.services.session_manager import SessionManager


def _ready_candidate(store: ChannelStore, message_id: str = "buyer-1") -> None:
    store.set_enabled("seller", True)
    message = InboundMessage(
        account_id="seller",
        platform_message_id=message_id,
        chat_id="chat-1",
        buyer_id="buyer-1",
        text="还在吗？",
    )
    assert store.record_inbound(message)
    claim = store.claim_for_generation("seller", message_id)
    assert claim is not None
    assert store.prepare_candidate(
        "seller",
        message_id,
        action="answer",
        candidate_text="还在。",
        account_version=claim["account_version"],
        session_version=claim["session_version"],
    ) is not None


def test_recovery_resumes_only_ready_candidates_with_unchanged_controls(tmp_path) -> None:
    store = ChannelStore(tmp_path / "channel.sqlite3")
    _ready_candidate(store)

    pending = store.pending_ready_deliveries("seller")
    assert pending == [{"platform_message_id": "buyer-1"}]
    delivery = store.claim_ready_delivery("seller", "buyer-1")

    assert delivery is not None
    assert delivery["candidate_text"] == "还在。"


def test_recovery_discards_ready_candidate_after_control_version_changes(tmp_path) -> None:
    store = ChannelStore(tmp_path / "channel.sqlite3")
    _ready_candidate(store)
    store.set_enabled("seller", False)
    store.set_enabled("seller", True)

    assert store.claim_ready_delivery("seller", "buyer-1") is None
    assert store.message("seller", "buyer-1")["status"] == "SUPERSEDED"


def test_recovery_marks_generation_and_inflight_send_without_repeating_them(tmp_path) -> None:
    store = ChannelStore(tmp_path / "channel.sqlite3")
    _ready_candidate(store, "sending-1")
    assert store.claim_ready_delivery("seller", "sending-1") is not None
    message = InboundMessage(
        account_id="seller",
        platform_message_id="generating-1",
        chat_id="chat-1",
        buyer_id="buyer-1",
        text="有没有维修过？",
    )
    assert store.record_inbound(message)
    assert store.claim_for_generation("seller", "generating-1") is not None

    result = store.recover_interrupted("seller")

    assert result == {"generation_interrupted": 1, "send_unknown": 1}
    assert store.message("seller", "generating-1")["status"] == "INTERRUPTED"
    sending = store.message("seller", "sending-1")
    assert sending["delivery_state"] == "UNKNOWN"
    assert store.claim_ready_delivery("seller", "sending-1") is None


def test_chat_service_total_deadline_includes_planning(tmp_path, monkeypatch) -> None:
    class SlowPlanner:
        async def plan_async(self, *_args, **_kwargs):
            await asyncio.sleep(0.1)

    monkeypatch.setattr(
        chat_service_module,
        "settings",
        SimpleNamespace(
            xianyu_expert_budget_seconds=0.01,
            xianyu_chat_api_timeout_seconds=0.1,
        ),
    )
    service = ChatService(
        planner=SlowPlanner(),
        session_manager=SessionManager(database_path=tmp_path / "sessions.sqlite3"),
    )

    result = asyncio.run(service.chat_async("还在吗？", "deadline-test"))

    assert result["reason"] == "turn_budget_exhausted"
    assert result["action"] == "error"
