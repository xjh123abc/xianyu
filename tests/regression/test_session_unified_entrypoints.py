from __future__ import annotations

import pytest

from app.services.chat_contracts import default_negotiation_state
from app.services.session_manager import SessionManager


def test_load_maps_existing_session_state_to_the_unified_context() -> None:
    manager = SessionManager()
    manager.append_turn(
        "chat-1",
        "查 TEST1001",
        "已找到",
        order_id="TEST1001",
        last_intent="order_query",
    )
    manager.set_current_item_id("chat-1", "item-001")
    manager.update_xianyu_context("chat-1", recent_price_topic="minimum")

    context = manager.load("chat-1")

    assert context.history[-1] == {"role": "assistant", "content": "已找到"}
    assert context.current_item_id == "ITEM-001"
    assert context.current_order_id == "TEST1001"
    assert context.last_task_type is None
    assert context.negotiation == {
        **default_negotiation_state(),
        "item_id": "ITEM-001",
    }
    assert context.platform_context["xianyu"] == {
        "item_id": "ITEM-001",
        "recent_price_topic": "minimum",
        "shipping_condition": None,
    }


def test_save_turn_persists_unified_context_through_sqlite(tmp_path) -> None:
    database_path = tmp_path / "sessions.sqlite3"
    first = SessionManager(database_path=database_path)
    first.save_turn(
        "chat-1",
        "最低多少？",
        "1490 元可以拍。",
        current_item_id="ITEM-001",
        current_order_id="TEST1001",
        last_task_type="price",
        legacy_intent="bargain",
        xianyu_context_updates={"recent_price_topic": "minimum"},
        negotiation={
            "round": 1,
            "last_ai_offer": 149000,
            "last_buyer_offer": 145000,
        },
    )

    restored = SessionManager(database_path=database_path).load("chat-1")

    assert restored.history == [
        {"role": "user", "content": "最低多少？"},
        {"role": "assistant", "content": "1490 元可以拍。"},
    ]
    assert restored.current_item_id == "ITEM-001"
    assert restored.current_order_id == "TEST1001"
    assert restored.last_task_type == "price"
    assert restored.platform_context["xianyu"]["recent_price_topic"] == "minimum"
    assert restored.negotiation == {
        "item_id": "ITEM-001",
        "round": 1,
        "last_ai_offer": 149000,
        "last_buyer_offer": 145000,
    }


def test_switching_items_resets_negotiation_state() -> None:
    manager = SessionManager()
    manager.save_turn(
        "chat-1",
        "1450 可以吗？",
        "1490 元可以拍。",
        current_item_id="ITEM-001",
        negotiation={"round": 2, "last_ai_offer": 149000, "last_buyer_offer": 145000},
    )
    manager.save_turn(
        "chat-1",
        "另一台多少钱？",
        "这台标价 1280 元。",
        current_item_id="ITEM-002",
    )

    assert manager.load("chat-1").negotiation == {
        **default_negotiation_state(),
        "item_id": "ITEM-002",
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"last_task_type": "unknown"}, "last_task_type"),
        ({"negotiation": {"round": -1}}, "negotiation.round"),
        ({"xianyu_context_updates": {"unexpected": "value"}}, "Unsupported Xianyu"),
    ],
)
def test_save_turn_rejects_invalid_unified_state(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SessionManager().save_turn("chat-1", "q", "a", **kwargs)


def test_invalid_save_turn_does_not_persist_a_partial_history_entry() -> None:
    manager = SessionManager()

    with pytest.raises(ValueError, match="negotiation.round"):
        manager.save_turn("chat-1", "q", "a", negotiation={"round": -1})

    assert manager.load("chat-1").history == []
