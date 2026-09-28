"""T1 R01/R02 integration coverage for the real unified chat path."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock

from fastapi.testclient import TestClient

from app.api import chat as chat_api
from app.api import chat_delivery
from app.main import app
from app.services.chat_service import ChatService
from app.services.item_service import ItemService
from app.services.negotiation_policy import NegotiationPolicyStore
from app.services.session_manager import SessionManager
from app.services.xianyu.experts.price_agent import PriceAgent


class NoRag:
    def prepare(self, *args: object, **kwargs: object) -> dict[str, object]:
        raise AssertionError("T1 price cases must not enter RAG")


CURRENT_ITEM_ID = "TEST_CORE_ALIGNMENT_CAMERA"


def _item(item_id: str = CURRENT_ITEM_ID) -> dict[str, object]:
    """Use the current configured item, never the documentation-only fixture."""

    item = ItemService().get_item_info(item_id)
    assert item["found"] is True
    return item


def _service(database_path: Path) -> ChatService:
    item = _item()
    mcp = Mock()
    mcp.get_item_info = AsyncMock(return_value=item)
    return ChatService(
        mcp_service=mcp,
        xianyu_rag_service=NoRag(),
        generator=Mock(),
        session_manager=SessionManager(database_path=database_path),
        price_agent=PriceAgent(NegotiationPolicyStore()),
    )


async def _submit(service: ChatService, response: dict[str, object], state: str) -> None:
    receipt = await service.record_delivery(
        str(response["chat_id"]),
        turn_id=str(response["turn_id"]),
        proposal_id=str(response["proposal_id"]),
        delivery_state=state,
    )
    assert receipt["accepted"] is True


def test_t1_holds_current_offer_on_generic_followup_and_confirms_quote(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path / "sessions.sqlite3")

    first = asyncio.run(
        service.chat_async(
            "能便宜一点吗？", "t1-three-tier", item_id=CURRENT_ITEM_ID, turn_id="m-1"
        )
    )
    assert first["answer"] == "你想多少收？整套机带镜头一起出，性价比已经挺高了。"
    assert "proposal_id" not in first

    offer = asyncio.run(service.chat_async("1450可以吗？", "t1-three-tier", turn_id="m-2"))
    assert offer["answer"] == (
        "¥1450.00 暂时不行，最多先比标价少10元包邮。"
        "整套机带镜头一起出，性价比已经挺高了。"
    )
    assert {"proposal_id", "turn_id"} <= offer.keys()
    assert "1400" not in str(offer)
    asyncio.run(_submit(service, offer, "LOCAL_SUBMITTED"))

    second = asyncio.run(service.chat_async("再少一点呢？", "t1-three-tier", turn_id="m-3"))
    assert second["answer"] == (
        "当前已经比标价少10元包邮，这次就不再往下调了。"
        "整套机带镜头一起出，性价比已经挺高了。"
    )
    assert "proposal_id" not in second

    confirmed = asyncio.run(
        service.chat_async("就按你刚才说的价格。", "t1-three-tier", turn_id="m-4")
    )
    assert confirmed["answer"] == "可以，就按 ¥1490.00包邮，目前仅确认报价，尚未实际改价或创建订单。"
    assert "proposal_id" not in confirmed
    state = service.session_manager.load("t1-three-tier").negotiation
    assert state["round"] == 1
    assert state["last_ai_offer"] == 149000
    assert state["shipping_condition"] == "seller_pays"
    assert state["offer_status"] == "confirmed"


def test_t1_same_message_is_idempotent_and_failed_or_unknown_delivery_does_not_advance(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path / "sessions.sqlite3")
    first = asyncio.run(
        service.chat_async(
            "1450可以吗？", "t1-idempotent", item_id=CURRENT_ITEM_ID, turn_id="m-1"
        )
    )
    replay = asyncio.run(
        service.chat_async(
            "1450可以吗？", "t1-idempotent", item_id=CURRENT_ITEM_ID, turn_id="m-1"
        )
    )
    assert replay["proposal_id"] == first["proposal_id"]
    assert replay["answer"] == first["answer"]
    asyncio.run(_submit(service, first, "FAILED"))
    state = service.session_manager.load("t1-idempotent").negotiation
    assert state["round"] == 0
    assert state["last_ai_offer"] is None
    assert state["pending_offer"] is None

    retry = asyncio.run(
        service.chat_async("1450可以吗？", "t1-idempotent", turn_id="m-2")
    )
    asyncio.run(_submit(service, retry, "UNKNOWN"))
    blocked = asyncio.run(
        service.chat_async("再少一点呢？", "t1-idempotent", turn_id="m-3")
    )
    assert "待确认" in str(blocked["answer"])
    assert service.session_manager.load("t1-idempotent").negotiation["round"] == 0


def test_t1_shipping_comparison_buyer_offer_and_restart_are_isolated(tmp_path: Path) -> None:
    database_path = tmp_path / "sessions.sqlite3"
    service = _service(database_path)
    comparison = asyncio.run(
        service.chat_async(
            "包邮最低多少？不包邮呢？", "t1-shipping", item_id=CURRENT_ITEM_ID, turn_id="m-1"
        )
    )
    assert str(comparison["answer"]).startswith(
        "包邮可以比标价少10元；不包邮可以比标价少30元。"
    )
    assert "proposal_id" not in comparison
    assert service.session_manager.load("t1-shipping").negotiation["round"] == 0

    accepted = asyncio.run(
        service.chat_async("不包邮，我出1475元可以吗？", "t1-shipping", turn_id="m-2")
    )
    assert accepted["answer"] == "可以，¥1475.00（不包邮）可以拍。"
    asyncio.run(_submit(service, accepted, "CONFIRMED"))
    restarted = _service(database_path)
    after_restart = restarted.session_manager.load("t1-shipping").negotiation
    assert after_restart["last_ai_offer"] == 147500
    assert after_restart["last_buyer_offer"] == 147500
    assert after_restart["shipping_condition"] == "buyer_pays"

    low = asyncio.run(
        restarted.chat_async("1300可以吗？", "t1-shipping", turn_id="m-3")
    )
    assert "暂时不行" in str(low["answer"])
    assert "1380" not in str(low)


def test_t1_http_chat_delivery_and_next_turn_use_one_session(
    tmp_path: Path, monkeypatch: object
) -> None:
    service = _service(tmp_path / "sessions.sqlite3")
    monkeypatch.setattr(chat_api, "chat_service", service)
    monkeypatch.setattr(chat_delivery, "chat_service", service)
    monkeypatch.setattr(chat_delivery.settings, "xianyu_delivery_token", "t1-token")

    with TestClient(app) as client:
        first = client.post(
            "/chat",
            json={
                "query": "1450可以吗？",
                "chat_id": "t1-http",
                "item_id": CURRENT_ITEM_ID,
                "turn_id": "http-m-1",
            },
        )
        assert first.status_code == 200
        first_body = first.json()
        receipt = client.post(
            "/internal/chat/delivery",
            json={
                "chat_id": "t1-http",
                "turn_id": first_body["turn_id"],
                "proposal_id": first_body["proposal_id"],
                "delivery_state": "CONFIRMED",
            },
            headers={"X-Internal-Token": "t1-token"},
        )
        second = client.post(
            "/chat",
            json={"query": "再少一点呢？", "chat_id": "t1-http", "turn_id": "http-m-2"},
        )

    assert first_body["answer"] == (
        "¥1450.00 暂时不行，最多先比标价少10元包邮。"
        "整套机带镜头一起出，性价比已经挺高了。"
    )
    assert receipt.json() == {"accepted": True, "idempotent": False}
    assert second.json()["answer"] == (
        "当前已经比标价少10元包邮，这次就不再往下调了。"
        "整套机带镜头一起出，性价比已经挺高了。"
    )


def test_t1_item_switch_and_expired_proposal_do_not_reuse_an_old_quote(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path / "sessions.sqlite3")
    first = asyncio.run(
        service.chat_async(
            "1450可以吗？", "t1-switch", item_id=CURRENT_ITEM_ID, turn_id="m-1"
        )
    )
    # Removing the currently loaded policy simulates a seller rule update:
    # the old opaque proposal must not be accepted after that change.
    service.price_agent.policy_store = NegotiationPolicyStore(tmp_path / "missing.json")
    expired = asyncio.run(
        service.record_delivery(
            "t1-switch",
            turn_id=str(first["turn_id"]),
            proposal_id=str(first["proposal_id"]),
            delivery_state="CONFIRMED",
        )
    )
    assert expired == {"accepted": False, "reason": "proposal_expired"}

    other = {
        **_item("DEMO_ITEM_001"),
        "title": "另一件测试商品",
        "listed_price_cents": 128000,
        "facts": {"negotiation": "firm", "shipping": {}},
    }

    async def _lookup(item_id: str) -> dict[str, object]:
        return _item() if item_id == CURRENT_ITEM_ID else other

    service.mcp_service.get_item_info = _lookup
    asyncio.run(
        service.chat_async(
            "这个商品多少钱？", "t1-switch", item_id="DEMO_ITEM_001", turn_id="m-2"
        )
    )
    state = service.session_manager.load("t1-switch").negotiation
    assert state["item_id"] == "DEMO_ITEM_001"
    assert state["round"] == 0
    assert state["last_ai_offer"] is None
