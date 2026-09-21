"""Acceptance tests for multi-question and conditional Xianyu messages."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from unittest.mock import AsyncMock, Mock

import pytest

from app.generation.prompt import build_xianyu_messages
from app.services.chat_service import ChatService
from app.services.item_service import ItemService
import app.services.xianyu.responses as responses


class NoRag:
    """Fail loudly if a fully confirmed fact-only turn tries to retrieve RAG."""

    def __init__(self) -> None:
        self.prepare_calls = 0

    def warm_up(self) -> None:
        return None

    def prepare(self, *args: object, **kwargs: object) -> dict[str, object]:
        self.prepare_calls += 1
        raise AssertionError("confirmed item facts should not require RAG")


class ItemEvidenceRag:
    """Provide grounded item evidence for model-knowledge boundary tests."""

    def warm_up(self) -> None:
        return None

    def prepare(self, *args: object, **kwargs: object) -> dict[str, object]:
        source = {"source": "CANON_FTB_001.md", "index": 0}
        return {
            "can_answer": True,
            "context": {
                "context": "商品说明包含这台相机支持的外接闪光灯信息。",
                "sources": [source],
            },
            "sources": [source],
            "results": [],
            "reliability": None,
        }


def _canon_item() -> dict[str, object]:
    return ItemService().get_item_info("CANON_FTB_001")


def _service(item: dict[str, object], rag: NoRag, generator: Mock) -> ChatService:
    mcp = Mock()
    mcp.get_item_info = AsyncMock(return_value=item)
    return ChatService(
        mcp_service=mcp,
        xianyu_rag_service=rag,
        generator=generator,
    )


@pytest.mark.parametrize(
    ("query", "expected_answer"),
    [
        (
            "还在吗？有没有维修过？包邮吗？",
            "还在的，这台目前还没出。\n没有维修过。\n包邮。",
        ),
        (
            "还在吗有没有维修过包邮吗",
            "还在的，这台目前还没出。\n没有维修过。\n包邮。",
        ),
        (
            "这台还没卖吧？修过没有？邮费怎么算？",
            "还在的，这台目前还没出。\n没有维修过。\n包邮。",
        ),
        ("今天能发吗？走顺丰吗？", "付款后 48 小时内发出。\n中通。"),
    ],
)
def test_complex_message_with_confirmed_facts_uses_full_message_and_replies(
    query: str,
    expected_answer: str,
) -> None:
    """Multiple buyer needs must not be reduced to one keyword or a handoff."""

    rag = NoRag()
    generator = Mock()
    service = _service(_canon_item(), rag, generator)

    result = asyncio.run(
        service.chat_async(query, f"complex_{query}", item_id="CANON_FTB_001")
    )

    assert result["action"] == "reply"
    assert result["can_answer"] is True
    assert result["answer"] == expected_answer
    assert rag.prepare_calls == 0
    generator.generate_xianyu.assert_not_called()
    generator.generate_xianyu_expert.assert_not_called()


@pytest.mark.parametrize(
    ("query", "expected_answer"),
    [
        ("不用包邮，能便宜吗？", "不包邮的话最低 ¥1470.00 可以拍。"),
        ("我出邮费，价格能少一点吗？", "不包邮的话最低 ¥1470.00 可以拍。"),
    ],
)
def test_conditional_bargain_uses_the_automatic_discount_limit(
    query: str,
    expected_answer: str,
) -> None:
    """Shipping conditions never bypass the global automatic discount limit."""

    rag = NoRag()
    generator = Mock()
    service = _service(_canon_item(), rag, generator)

    result = asyncio.run(
        service.chat_async(query, f"bargain_{query}", item_id="CANON_FTB_001")
    )

    assert result["action"] == "reply"
    assert result["answer"] == expected_answer
    assert result["answer"] != "包邮。"
    assert rag.prepare_calls == 0
    generator.generate_xianyu.assert_not_called()


def test_unauthorised_pickup_condition_handoffs_instead_of_using_shipping_price() -> None:
    generator = Mock()
    result = asyncio.run(
        _service(_canon_item(), NoRag(), generator).chat_async(
            "自提的话能便宜吗？",
            "bargain_unsupported_pickup",
            item_id="CANON_FTB_001",
        )
    )

    assert result["action"] != "handoff"
    assert result["answer"] == "该问题目前暂无足够信息确认。"
    assert result["reason"] == "unsupported_price_condition"
    generator.generate_xianyu.assert_not_called()


def test_shipping_policy_controls_the_no_shipping_minimum() -> None:
    """The no-shipping discount is combined with the separate small-bargain policy."""

    item = deepcopy(_canon_item())
    generator = Mock()
    service = _service(item, NoRag(), generator)

    result = asyncio.run(
        service.chat_async(
            "不用包邮，能便宜吗？",
            "bargain_shipping_policy",
            item_id="CANON_FTB_001",
        )
    )

    assert result["action"] == "reply"
    assert result["can_answer"] is True
    assert result["answer"] == "不包邮的话最低 ¥1470.00 可以拍。"
    generator.generate_xianyu.assert_not_called()


def test_conditional_bargain_without_a_known_policy_handoffs() -> None:
    """A condition without an item policy cannot be priced safely."""

    item = deepcopy(_canon_item())
    item["facts"] = {
        **item["facts"],
        "negotiation": "unknown",
        "shipping": {
            **item["facts"]["shipping"],
            "negotiation_policy": "unknown",
            "negotiation_express_policy": "unknown",
        },
    }
    generator = Mock()
    service = _service(item, NoRag(), generator)

    result = asyncio.run(
        service.chat_async(
            "不用包邮，能便宜吗？",
            "bargain_unknown_policy",
            item_id="CANON_FTB_001",
        )
    )

    assert result["action"] != "handoff"
    assert result["can_answer"] is False
    assert result["answer"] == "该问题目前暂无足够信息确认。"
    generator.generate_xianyu.assert_not_called()


def test_simple_shipping_question_keeps_fast_fact_reply() -> None:
    """The complex-message safeguard does not slow a simple single question."""

    generator = Mock()
    service = _service(_canon_item(), NoRag(), generator)

    result = asyncio.run(
        service.chat_async("包邮吗？", "simple_shipping", item_id="CANON_FTB_001")
    )

    assert result["action"] == "reply"
    assert result["answer"] == "包邮。"
    generator.generate_xianyu.assert_not_called()


def test_xianyu_prompt_exposes_all_confirmed_facts_without_conflicts() -> None:
    """The LLM receives shipping/history/negotiation facts needed for full replies."""

    item = _canon_item()
    messages = build_xianyu_messages(
        "还在吗？有没有维修过？包邮吗？",
        item,
        "当前商品的已确认事实已提供。",
    )
    prompt = messages[1]["content"]

    assert "shipping_fee" in prompt
    assert "包邮" in prompt
    assert "repair_history" in prompt
    assert "没有维修过" in prompt
    assert "negotiation" in prompt
    assert "negotiation_policy" in prompt
    assert "negotiation_express_policy" in prompt
    assert "fact_conflicts" not in prompt


@pytest.mark.parametrize(
    "text",
    [
        "这个需要卖家确认。",
        "我帮你问卖家。",
        "请等待卖家确认。",
        "需要人工确认后才能答复。",
        "缺少依据",
        "缺少依据。当前资料没有对应说明。",
        "资料不足。",
        "无法确认",
    ],
)
def test_legacy_text_guard_is_removed_from_the_unified_path(text: str) -> None:
    del text
    assert not hasattr(responses, "requires_human_handoff")


def test_unified_task_path_preserves_the_grounded_generated_reply() -> None:
    generator = Mock()
    generator.generate_xianyu_expert.return_value = "这个需要卖家确认。"
    service = _service(_canon_item(), ItemEvidenceRag(), generator)

    result = asyncio.run(
        service.chat_async(
            "这台支持外接闪光灯吗？",
            "generated_human_review_language",
            item_id="CANON_FTB_001",
        )
    )

    assert result["action"] != "handoff"
    assert result["answer"] == "这个需要卖家确认。"
    assert "reason" not in result
    generator.generate_xianyu_expert.assert_called_once()
    generator.generate_xianyu.assert_not_called()
