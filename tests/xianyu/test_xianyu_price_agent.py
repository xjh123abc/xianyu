"""S2 unit coverage for the deterministic Xianyu pricing expert."""

from __future__ import annotations

from copy import deepcopy
from unittest.mock import Mock

import pytest

from app.services.intent_router import IntentRouter
from app.services.item_service import ItemService
from app.services.xianyu.experts.contracts import PriceDecision
from app.services.xianyu.experts.price_agent import PriceAgent
from app.services.xianyu.item_fact_responder import ItemFactResponder


def _canon_item() -> dict[str, object]:
    return ItemService().get_item_info("CANON_FTB_001")


def _decision(query: str, item: dict[str, object] | None = None) -> PriceDecision:
    return PriceAgent().decide(query, item or _canon_item(), IntentRouter().route(query))


@pytest.mark.parametrize(
    ("query", "answer", "shipping_condition", "minimum_price_cents"),
    [
        (
            "包邮最低多少？",
            "可以先比标价少10元包邮。整套机带镜头一起出，性价比已经挺高了。",
            "seller_pays",
            140000,
        ),
        (
            "不包邮最低多少？",
            "可以先比标价少30元不包邮。整套机带镜头一起出，性价比已经挺高了。",
            "buyer_pays",
            138000,
        ),
        (
            "包邮最低多少？不包邮呢？",
            "包邮可以比标价少10元；不包邮可以比标价少30元。整套机带镜头一起出，性价比已经挺高了。",
            None,
            None,
        ),
        ("包邮1495可以吗？", "可以，¥1495.00包邮可以拍。", "seller_pays", 140000),
        ("不包邮我出1470元可以吗？", "可以，¥1470.00（不包邮）可以拍。", "buyer_pays", 138000),
    ],
)
def test_price_agent_calculates_the_single_authorised_policy(
    query: str,
    answer: str,
    shipping_condition: str | None,
    minimum_price_cents: int | None,
) -> None:
    decision = _decision(query)

    assert decision.status == "answered"
    assert decision.answer == answer
    assert decision.original_price_cents == 150000
    assert decision.minimum_price_cents == minimum_price_cents
    assert decision.shipping_condition == shipping_condition


@pytest.mark.parametrize(
    ("query", "reason"),
    [("自提的话能便宜吗？", "unsupported_price_condition"), ("不要镜头能便宜吗？", "unsupported_price_condition")],
)
def test_price_agent_handoffs_for_unauthorised_conditions(
    query: str,
    reason: str,
) -> None:
    decision = _decision(query)

    assert decision.status == "handoff"
    assert decision.answer is None
    assert decision.reason is not None
    assert reason in decision.reason


def test_private_policy_is_not_overridden_by_public_reference_price_text() -> None:
    item = deepcopy(_canon_item())
    item["facts"] = {
        **item["facts"],
        "shipping": {
            **item["facts"]["shipping"],
            "negotiation_policy": "小刀10元，1490元可以拍",
        },
    }

    decision = _decision("最低多少？", item)

    assert decision.status == "answered"
    assert decision.answer == "可以先比标价少10元包邮。整套机带镜头一起出，性价比已经挺高了。"


def test_price_agent_handoffs_on_fact_conflicts_but_ignores_public_policy_tampering() -> None:
    conflict = deepcopy(_canon_item())
    conflict["facts"] = {**conflict["facts"], "fact_conflicts": ["shipping"]}
    excessive = deepcopy(_canon_item())
    excessive["facts"] = {
        **excessive["facts"],
        "shipping": {
            **excessive["facts"]["shipping"],
            "negotiation_policy": "累计最多小刀2000元",
        },
    }

    conflict_decision = _decision("最低多少？", conflict)
    excessive_decision = _decision("最低多少？", excessive)

    assert conflict_decision.status == "handoff"
    assert conflict_decision.reason == "price_fact_conflict:shipping"
    assert excessive_decision.status == "answered"
    assert excessive_decision.answer == "可以先比标价少10元包邮。整套机带镜头一起出，性价比已经挺高了。"


def test_generic_discount_question_asks_for_buyer_budget_first() -> None:
    decision = _decision("能便宜点吗？")

    assert decision.status == "answered"
    assert decision.answer == "你想多少收？整套机带镜头一起出，性价比已经挺高了。"
    assert decision.state_proposal is None
    assert decision.effective_price_cents is None


@pytest.mark.parametrize("query", ["这个相机最低多少1450可以吗", "1450可以吗？"])
def test_buyer_offer_below_current_tier_is_not_accepted(query: str) -> None:
    decision = _decision(query)

    assert decision.status == "answered"
    assert decision.answer == (
        "¥1450.00 暂时不行，最多先比标价少10元包邮。"
        "整套机带镜头一起出，性价比已经挺高了。"
    )
    assert decision.buyer_offer_cents == 145000
    assert decision.effective_price_cents == 149000


def test_price_agent_does_not_accept_a_sold_item() -> None:
    item = deepcopy(_canon_item())
    item["sale_status"] = "sold"

    decision = _decision("我出1495元可以吗？", item)

    assert decision.status == "answered"
    assert decision.answer == "这件已经出掉了。"
    assert decision.minimum_price_cents is None


def test_price_agent_recalculates_from_the_original_price_on_every_turn() -> None:
    first = _decision("不包邮最低多少？")
    second = _decision("不包邮最低多少？")

    assert first.minimum_price_cents == 138000
    assert second.minimum_price_cents == 138000
    assert first.answer == second.answer


def test_item_fact_responder_only_maps_the_price_expert_decision_to_a_reply() -> None:
    price_agent = Mock()
    price_agent.decide.return_value = PriceDecision.answered("可以，¥1495.00 可以拍。")
    item = _canon_item()
    responder = ItemFactResponder(price_agent=price_agent)
    match = IntentRouter().route("我出1495元可以吗？")

    response = responder.answer_intent("我出1495元可以吗？", item, match)

    price_agent.decide.assert_called_once_with("我出1495元可以吗？", item, match)
    assert response["action"] == "reply"
    assert response["answer"] == "可以，¥1495.00 可以拍。"
