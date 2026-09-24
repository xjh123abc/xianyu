"""Stage 3 checks for seller-maintained structured product facts."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from unittest.mock import AsyncMock, Mock

import pytest

from app.services.chat_service import ChatService
from app.generation.prompt import build_xianyu_messages
from app.services.item_service import ItemService
from app.services.session_manager import SessionManager


class NoRag:
    def __init__(self) -> None:
        self.item_ids: list[str | None] = []

    def warm_up(self) -> None:
        return None

    def prepare(self, query: str, *, item_id: str | None = None) -> dict[str, object]:
        self.item_ids.append(item_id)
        return {
            "can_answer": False,
            "context": None,
            "sources": [],
            "results": [],
            "reliability": None,
        }


def _service(item: dict[str, object], rag: NoRag) -> ChatService:
    mcp = Mock()
    mcp.get_item_info = AsyncMock(return_value=item)
    return ChatService(
        mcp_service=mcp,
        xianyu_rag_service=rag,
        generator=Mock(),
        session_manager=SessionManager(),
    )


def _canon_item() -> dict[str, object]:
    return ItemService().get_item_info("CANON_FTB_001")


def test_item_service_preserves_public_seller_structured_facts() -> None:
    item = _canon_item()

    assert item["found"] is True
    assert item["facts"]["identity"] == {
        "category": "胶片相机", "brand": "Canon", "model": "FTb"
    }
    assert item["facts"]["condition"]["appearance"] == "正常使用痕迹"
    assert item["facts"]["condition"]["summary"] == "机身九成新，有轻微使用痕迹"
    assert item["facts"]["function"] == {"overall": "working", "shutter": "working"}
    assert item["facts"]["history"]["repair_history"] == "没有维修过"
    assert item["facts"]["lens"]["model"] == "FD 50mm F1.8"
    assert item["facts"]["included_items"] == ["Canon FTb 机身", "50mm 镜头", "镜头盖"]
    assert item["facts"]["shipping"]["shipping_fee"] == "包邮"
    assert item["facts"]["shipping"]["negotiation_policy"] == "累计最多小刀100元"
    assert item["facts"]["shipping"]["negotiation_express_policy"] == "不包邮商品价减20元，可叠加小刀"
    assert item["facts"]["listing_description"]
    assert "platform_item_id" not in item
    assert ItemService().resolve_item_ids("我想问 CANON_FTB_001") == ["CANON_FTB_001"]


def test_confirmed_lens_is_answered_without_rag_or_llm() -> None:
    rag = NoRag()
    service = _service(_canon_item(), rag)

    result = asyncio.run(
        service.chat_async("这台带的镜头是什么焦段？", "stage3_known_lens", item_id="CANON_FTB_001")
    )

    assert result["action"] == "reply"
    assert result["can_answer"] is True
    assert "FD 50mm F1.8" in str(result["answer"])
    assert "50mm" in str(result["answer"])
    assert "卖家资料" not in str(result["answer"])
    assert rag.item_ids == []
    service.generator.generate.assert_not_called()


def test_confirmed_status_uses_a_brief_customer_service_reply() -> None:
    rag = NoRag()
    service = _service(_canon_item(), rag)

    result = asyncio.run(
        service.chat_async("这台还在吗？", "stage4_known_status", item_id="CANON_FTB_001")
    )

    assert result["action"] == "reply"
    assert result["answer"] == "还在的，这台目前还没出。"
    assert "卖家资料" not in str(result["answer"])
    assert "CANON_FTB_001" not in str(result["answer"])
    assert rag.item_ids == []


@pytest.mark.parametrize(
    ("query", "expected", "internal_label"),
    [
        ("包邮吗？", "包邮。", "运费："),
        ("有维修过吗？", "没有维修过。", "维修历史："),
        ("多久发货？", "付款后 48 小时内发出。", "发货时间："),
        ("成色怎么样？", "机身九成新，有轻微使用痕迹。", "成色情况："),
    ],
)
def test_keyword_fact_reply_hides_internal_storage_labels(
    query: str,
    expected: str,
    internal_label: str,
) -> None:
    """Direct fact replies expose seller wording only, never a field heading."""

    result = asyncio.run(
        _service(_canon_item(), NoRag()).chat_async(
            query,
            f"natural_reply_{query}",
            item_id="CANON_FTB_001",
        )
    )

    assert result["action"] == "reply"
    assert result["answer"] == expected
    assert internal_label not in str(result["answer"])


def test_xianyu_prompt_has_context_but_no_internal_item_id() -> None:
    messages = build_xianyu_messages(
        "这个带什么配件？",
        _canon_item(),
        "商品说明只确认了镜头盖。",
        [{"role": "user", "content": "这台还在吗？"}],
    )

    combined = "\n".join(message["content"] for message in messages)
    assert "CANON_FTB_001" not in combined
    assert "1084130180117" not in combined
    assert "这台还在吗？" in combined
    assert "镜头盖" in combined


def test_confirmed_condition_and_included_items_are_answered_from_facts() -> None:
    rag = NoRag()
    service = _service(_canon_item(), rag)

    result = asyncio.run(
        service.chat_async(
            "这台成色怎么样，带哪些配件？",
            "stage3_known_condition_and_items",
            item_id="CANON_FTB_001",
        )
    )

    assert result["action"] == "reply"
    assert result["answer"] == (
        "机身九成新，有轻微使用痕迹。\n"
        "一起出的有 Canon FTb 机身、FD 50mm F1.8 镜头、镜头盖。"
    )
    assert rag.item_ids == []
    service.generator.generate_xianyu.assert_not_called()
    service.generator.generate_xianyu_expert.assert_not_called()


def test_confirmed_brand_and_model_are_answered_from_facts() -> None:
    rag = NoRag()
    service = _service(_canon_item(), rag)

    result = asyncio.run(
        service.chat_async("这是什么型号？", "stage3_known_model", item_id="CANON_FTB_001")
    )

    assert result["action"] == "reply"
    assert "Canon" in str(result["answer"])
    assert "FTb" in str(result["answer"])
    assert rag.item_ids == []


def test_explicitly_unknown_structured_fact_handoffs_without_guessing() -> None:
    item = deepcopy(_canon_item())
    item["facts"] = {"lens": None}
    rag = NoRag()

    result = asyncio.run(
        _service(item, rag).chat_async(
            "这台带什么镜头？", "stage3_unknown_lens", item_id="CANON_FTB_001"
        )
    )

    assert result["action"] != "handoff"
    assert result["can_answer"] is False
    assert result["answer"] == ""
    assert "50mm" not in str(result["answer"])
    assert rag.item_ids == []


def test_explicit_fact_conflict_handoffs_without_selecting_a_value() -> None:
    item = deepcopy(_canon_item())
    item["facts"] = {**item["facts"], "fact_conflicts": ["lens"]}
    rag = NoRag()

    result = asyncio.run(
        _service(item, rag).chat_async(
            "这台带什么镜头？", "stage3_conflicting_lens", item_id="CANON_FTB_001"
        )
    )

    assert result["action"] != "handoff"
    assert result["can_answer"] is False
    assert result["answer"] == ""
    assert "50mm" not in str(result["answer"])
    assert rag.item_ids == []


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("我出1490元可以吗？", "可以，¥1490.00包邮可以拍。"),
        ("我出1495元能出吗？", "可以，¥1495.00包邮可以拍。"),
        ("1490元可以吗？", "可以，¥1490.00包邮可以拍。"),
        ("能便宜点吗？", "你想多少收？整套机带镜头一起出，性价比已经挺高了。"),
    ],
)
def test_bargain_within_automatic_limit_replies_from_original_price(
    query: str,
    expected: str,
) -> None:
    result = asyncio.run(
        _service(_canon_item(), NoRag()).chat_async(
            query,
            f"automatic_bargain_{query}",
            item_id="CANON_FTB_001",
        )
    )

    assert result["action"] == "reply"
    assert result["answer"] == expected


def test_bargain_below_authorised_floor_replies_without_revealing_the_floor() -> None:
    result = asyncio.run(
        _service(_canon_item(), NoRag()).chat_async(
            "我出1399元可以吗？",
            "automatic_bargain_below_limit",
            item_id="CANON_FTB_001",
        )
    )

    assert result["action"] != "handoff"
    assert result["can_answer"] is True
    assert result["answer"] == (
        "¥1399.00 暂时不行，最多先比标价少10元包邮。"
        "整套机带镜头一起出，性价比已经挺高了。"
    )
    assert "1400" not in str(result["answer"])


def test_no_shipping_policy_applies_before_the_minor_bargain_discount() -> None:
    result = asyncio.run(
        _service(_canon_item(), NoRag()).chat_async(
            "最低多少？不包邮的话最低多少？",
            "no_shipping_lowest_price",
            item_id="CANON_FTB_001",
        )
    )

    assert result["action"] == "reply"
    assert result["can_answer"] is True
    assert result["answer"] == (
        "包邮可以比标价少10元；不包邮可以比标价少30元。"
        "整套机带镜头一起出，性价比已经挺高了。"
    )


def test_offer_below_known_no_shipping_floor_repeats_current_tier() -> None:
    result = asyncio.run(
        _service(_canon_item(), NoRag()).chat_async(
            "不包邮，我出1370元可以吗？",
            "no_shipping_offer_below_minimum",
            item_id="CANON_FTB_001",
        )
    )

    assert result["action"] != "handoff"
    assert result["answer"] == (
        "¥1370.00 暂时不行，最多先比标价少30元不包邮。"
        "整套机带镜头一起出，性价比已经挺高了。"
    )
    assert "1380" not in str(result["answer"])


def test_shipping_included_offer_below_floor_repeats_current_tier() -> None:
    result = asyncio.run(
        _service(_canon_item(), NoRag()).chat_async(
                "1390包邮可以吗？",
            "shipping_included_offer_below_minimum",
            item_id="CANON_FTB_001",
        )
    )

    assert result["action"] != "handoff"
    assert result["answer"] == (
        "¥1390.00 暂时不行，最多先比标价少10元包邮。"
        "整套机带镜头一起出，性价比已经挺高了。"
    )
    assert "1400" not in str(result["answer"])


def test_generic_followup_holds_current_offer_after_delivery_receipt() -> None:
    rag = NoRag()
    service = _service(_canon_item(), rag)

    first = asyncio.run(
        service.chat_async("不包邮最低多少？", "automatic_bargain_no_accumulation", item_id="CANON_FTB_001")
    )
    assert first["answer"] == "可以先比标价少30元不包邮。整套机带镜头一起出，性价比已经挺高了。"
    asyncio.run(
        service.record_delivery(
            "automatic_bargain_no_accumulation",
            turn_id=str(first["turn_id"]),
            proposal_id=str(first["proposal_id"]),
            delivery_state="CONFIRMED",
        )
    )
    second = asyncio.run(
        service.chat_async("还能再便宜 10 元吗？", "automatic_bargain_no_accumulation", item_id="CANON_FTB_001")
    )

    assert second["action"] == "reply"
    assert second["answer"] == (
        "当前已经比标价少30元不包邮，这次就不再往下调了。"
        "整套机带镜头一起出，性价比已经挺高了。"
    )
    assert rag.item_ids == []
    service.generator.generate_xianyu.assert_not_called()
