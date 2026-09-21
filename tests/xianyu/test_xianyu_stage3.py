"""Stage 3 acceptance tests for independent MCP/RAG evidence handling."""

from __future__ import annotations

import asyncio
from uuid import uuid4
from unittest.mock import AsyncMock, Mock

from app.services.chat_service import ChatService
from app.services.item_service import ItemService
from app.services.query_planner import build_question_plan


class EvidenceRag:
    def __init__(self, *, can_answer: bool = True) -> None:
        self.can_answer = can_answer
        self.item_ids: list[str | None] = []
        self.queries: list[str] = []

    def warm_up(self) -> None:
        return None

    def prepare(self, query: str, *, item_id: str | None = None) -> dict[str, object]:
        self.queries.append(query)
        self.item_ids.append(item_id)
        return {
            "can_answer": self.can_answer,
            "context": {
                "context": "已确认付款后，卖家通常会在 48 小时内安排发出。",
                "sources": [{"source": "seller_rules.md", "index": 0}],
            }
            if self.can_answer
            else None,
            "sources": [{"source": "seller_rules.md", "index": 0}]
            if self.can_answer
            else [],
            "results": [],
            "reliability": {
                "can_answer": self.can_answer,
                "next_step": "context_builder" if self.can_answer else "human_handoff",
                "reason": "sufficient_evidence" if self.can_answer else "no_results",
                "top_rerank_score": 0.9 if self.can_answer else None,
                "threshold": 0.35,
            },
        }


def _service(rag: EvidenceRag, item_lookup: AsyncMock | None = None) -> ChatService:
    mcp = Mock()
    mcp.get_item_info = item_lookup or AsyncMock(
        side_effect=lambda item_id: ItemService().get_item_info(item_id)
    )
    generator = Mock()
    generator.generate_xianyu_expert.return_value = (
        "根据卖家规则，已确认付款后通常会在 48 小时内安排发出。"
    )
    return ChatService(
        mcp_service=mcp,
        xianyu_rag_service=rag,
        generator=generator,
    )


def test_plan_keeps_item_price_and_common_knowledge_as_separate_needs() -> None:
    plan = build_question_plan("这个商品多少钱？收到有质量问题怎么办？")

    assert plan["item_fields"] == ["listed_price_cents"]
    assert plan["knowledge_questions"] == [
        {
            "question": "收到有质量问题怎么办",
            "scope": "common",
        }
    ]


def test_shipping_fee_is_not_misclassified_as_item_price() -> None:
    plan = build_question_plan("退货运费多少钱？")

    assert plan["item_fields"] == []
    assert plan["knowledge_questions"][0]["scope"] == "common"


def test_logistics_status_is_not_misclassified_as_item_sale_status() -> None:
    plan = build_question_plan("这个订单的物流状态是什么？")

    assert "sale_status" not in plan["item_fields"]


def test_can_still_buy_is_classified_as_item_sale_status() -> None:
    assert build_question_plan("这个商品还能买吗？")["item_fields"] == ["sale_status"]


def test_seller_general_question_uses_common_knowledge_without_mcp() -> None:
    rag = EvidenceRag()
    service = _service(rag)

    result = asyncio.run(
        service.chat_async("你们店一般怎么处理售后？", "stage3_general")
    )

    assert result["can_answer"] is True
    assert result["sources"] == [{"source": "seller_rules.md", "index": 0}]
    assert rag.item_ids == [None]
    service.mcp_service.get_item_info.assert_not_awaited()
    service.generator.generate_xianyu_expert.assert_called_once()
    service.generator.generate_xianyu.assert_not_called()


def test_unconfirmed_return_promise_uses_current_common_corpus() -> None:
    rag = EvidenceRag()
    service = _service(rag)

    result = asyncio.run(
        service.chat_async("你们店支持七天无理由退货吗？", "stage3_old_promise")
    )

    assert result["route"] == "xianyu"
    assert result["sources"] == [{"source": "seller_rules.md", "index": 0}]
    assert "卖家通用规则" in rag.queries[0]
    service.mcp_service.get_item_info.assert_not_awaited()


def test_unknown_status_handoff_keeps_mcp_source() -> None:
    service = _service(EvidenceRag())

    result = asyncio.run(
        service.chat_async(
            "这个商品还在售吗？",
            "stage3_unknown_status_source",
            item_id="DEMO_ITEM_003",
        )
    )

    assert result["sources"] == [
        {"source": "mcp:get_item_info", "index": "DEMO_ITEM_003"}
    ]


def test_mcp_failure_keeps_independent_common_knowledge_answer() -> None:
    rag = EvidenceRag()
    item_lookup = AsyncMock(side_effect=RuntimeError("MCP unavailable"))
    service = _service(rag, item_lookup)

    result = asyncio.run(
        service.chat_async(
            "这个商品多少钱？收到有质量问题怎么办？",
            "stage3_mcp_failure",
            item_id="DEMO_ITEM_001",
        )
    )

    assert result["can_answer"] is False
    assert result["next_step"] != "human_handoff"
    assert result["answer"] == "该问题目前暂无足够信息确认。"
    assert result["sources"] == [{"source": "seller_rules.md", "index": 0}]
    item_lookup.assert_awaited_once_with("DEMO_ITEM_001")
    assert rag.item_ids == [None]


def test_missing_item_keeps_common_answer_and_asks_for_item() -> None:
    rag = EvidenceRag()
    service = _service(rag)

    result = asyncio.run(
        service.chat_async(
            "多少钱？收到有质量问题怎么办？",
            "stage3_missing_item",
        )
    )

    assert result["can_answer"] is False
    assert result["next_step"] != "human_handoff"
    assert result["answer"] == "该问题目前暂无足够信息确认。"
    assert "DEMO_ITEM_001" not in str(result["answer"])
    service.mcp_service.get_item_info.assert_not_awaited()
    assert rag.item_ids == [None]


def test_unknown_status_does_not_fall_back_to_common_shipping_rules() -> None:
    rag = EvidenceRag()
    service = _service(rag)

    result = asyncio.run(
        service.chat_async(
            "现在还在售吗？一般多久发货？",
            "stage3_unknown_status",
            item_id="DEMO_ITEM_003",
        )
    )

    assert result["can_answer"] is False
    assert result["next_step"] != "human_handoff"
    assert result["answer"] == "该问题目前暂无足够信息确认。"
    assert result["sources"] == [
        {"source": "mcp:get_item_info", "index": "DEMO_ITEM_003"}
    ]
    assert rag.item_ids == []


def test_structured_subquestions_do_not_fall_back_to_item_knowledge() -> None:
    rag = EvidenceRag()
    original_prepare = rag.prepare

    def fail_item_scope_only(
        query: str,
        *,
        item_id: str | None = None,
    ) -> dict[str, object]:
        prepared = original_prepare(query, item_id=item_id)
        if item_id is not None:
            prepared.update({"can_answer": False, "context": None, "sources": []})
        return prepared

    rag.prepare = fail_item_scope_only  # type: ignore[method-assign]
    service = _service(rag)

    result = asyncio.run(
        service.chat_async(
            "多少钱？有什么配件？你们店一般怎么处理售后？",
            "stage3_partial_knowledge",
            item_id="DEMO_ITEM_001",
        )
    )

    assert result["can_answer"] is True
    assert result["answer"] == (
        "这件标价是 ¥1280.00。\n"
        "一起出的有 相机机身、相机背带、镜头盖。\n"
        "根据卖家规则，已确认付款后通常会在 48 小时内安排发出。"
    )
    assert rag.item_ids == [None]
    expert, question, item, evidence = service.generator.generate_xianyu_expert.call_args.args
    assert expert == "service"
    assert question == "售后或店铺通用规则"
    assert item is not None
    assert "1280.00" not in evidence
    service.generator.generate_xianyu.assert_not_called()
    assert result["sources"] == [
        {"source": "mcp:get_item_info", "index": "DEMO_ITEM_001"},
        {"source": "seller_rules.md", "index": 0},
    ]


def test_pure_item_fact_has_tool_source_and_does_not_call_rag() -> None:
    rag = EvidenceRag()
    service = _service(rag)

    result = asyncio.run(
        service.chat_async(
            "这个商品多少钱？",
            "stage3_price",
            item_id="DEMO_ITEM_001",
        )
    )

    assert result["can_answer"] is True
    assert "1280.00" in str(result["answer"])
    assert result["sources"] == [
        {"source": "mcp:get_item_info", "index": "DEMO_ITEM_001"}
    ]
    assert rag.queries == []
    service.mcp_service.get_item_info.assert_awaited_once_with("DEMO_ITEM_001")


def test_mismatched_mcp_item_is_rejected_and_not_saved() -> None:
    rag = EvidenceRag()
    mismatched = ItemService().get_item_info("DEMO_ITEM_002")
    service = _service(rag, AsyncMock(return_value=mismatched))

    result = asyncio.run(
        service.chat_async(
            "这个商品多少钱？",
            "stage3_mismatch",
            item_id="DEMO_ITEM_001",
        )
    )

    assert result["can_answer"] is False
    assert result["next_step"] != "human_handoff"
    assert result["answer"] == "该问题目前暂无足够信息确认。"
    assert service.session_manager.get_current_item_id("stage3_mismatch") is None


def test_knowledge_without_a_valid_source_cannot_make_a_positive_promise() -> None:
    rag = EvidenceRag()
    original_prepare = rag.prepare

    def prepare_without_sources(
        query: str,
        *,
        item_id: str | None = None,
    ) -> dict[str, object]:
        prepared = original_prepare(query, item_id=item_id)
        prepared["sources"] = []
        return prepared

    rag.prepare = prepare_without_sources  # type: ignore[method-assign]
    service = _service(rag)

    result = asyncio.run(service.chat_async("你们店一般多久发货？", "stage3_no_source"))

    assert result["can_answer"] is False
    assert result["next_step"] != "human_handoff"
    assert result["answer"] == "该问题目前暂无足够信息确认。"


def test_common_generation_failure_keeps_retrieved_evidence() -> None:
    rag = EvidenceRag()
    service = _service(rag)
    service.generator.generate_xianyu_expert.side_effect = RuntimeError("empty answer")

    result = asyncio.run(
        service.chat_async("你们店售后怎么处理？", "stage3_common_failure")
    )

    assert result["can_answer"] is False
    assert result["next_step"] != "human_handoff"
    assert result["answer"] == "该问题目前暂无足够信息确认。"
    assert result["sources"] == [{"source": "seller_rules.md", "index": 0}]
    service.generator.generate_xianyu.assert_not_called()


def test_explicit_item_routes_unlisted_damage_question_to_item_knowledge() -> None:
    rag = EvidenceRag()
    service = _service(rag)

    result = asyncio.run(
        service.chat_async(
            "这台相机以前有没有摔过？",
            "stage3_test_005",
            item_id="DEMO_ITEM_001",
        )
    )

    assert result["action"] != "handoff"
    assert result["item_id"] == "DEMO_ITEM_001"
    assert rag.item_ids == []
    assert service.session_manager.get_current_item_id("stage3_test_005") == "DEMO_ITEM_001"


def test_explicit_item_limits_usb_cable_question_to_selected_item() -> None:
    rag = EvidenceRag()
    original_prepare = rag.prepare

    def selected_item_sources(
        query: str,
        *,
        item_id: str | None = None,
    ) -> dict[str, object]:
        prepared = original_prepare(query, item_id=item_id)
        prepared["sources"] = [
            {"source": "DEMO_ITEM_001.md", "index": 0},
            {"source": "seller_rules.md", "index": 0},
        ]
        return prepared

    rag.prepare = selected_item_sources  # type: ignore[method-assign]
    service = _service(rag)

    result = asyncio.run(
        service.chat_async(
            "这个商品带USB数据线吗？",
            "stage3_test_006",
            item_id="DEMO_ITEM_001",
        )
    )

    assert result["action"] != "clarify"
    assert result["item_id"] == "DEMO_ITEM_001"
    assert rag.item_ids == ["DEMO_ITEM_001"]
    assert {source["source"] for source in result["sources"]} == {
        "mcp:get_item_info",
        "DEMO_ITEM_001.md",
        "seller_rules.md",
    }
    assert all("DEMO_ITEM_002.md" not in source["source"] for source in result["sources"])


def test_item_question_without_request_or_memory_item_handoffs() -> None:
    for index, query in enumerate(
        ("这台相机以前有没有摔过？", "这个商品带USB数据线吗？")
    ):
        service = _service(EvidenceRag())

        result = asyncio.run(
            service.chat_async(query, f"stage3_missing_{index}_{uuid4().hex}")
        )

        assert result["action"] != "handoff"
        assert result["next_step"] != "human_handoff"
        assert result["answer"] == "请补充商品编号或具体商品信息。"
        service.mcp_service.get_item_info.assert_not_awaited()
