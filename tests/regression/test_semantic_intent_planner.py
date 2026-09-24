"""S2.5A semantic-understanding planner regression coverage."""

from __future__ import annotations

import asyncio

from app.services.chat_contracts import SessionContext, TaskResult
from app.services.intent_router import IntentRouter
from app.services.planner import Planner
from app.services.result_merger import ResultMerger


def _planner() -> Planner:
    return Planner(
        intent_router=IntentRouter(),
        requires_item_context=lambda query: True,
        may_contain_explicit_item_reference=lambda query: "ITEM-" in query,
    )


def _plan(query: str, context: SessionContext | None = None):
    return asyncio.run(
        _planner().plan_async(
            query,
            context or SessionContext(current_item_id="CANON_FTB_001"),
            item_id="CANON_FTB_001",
        )
    )


def test_hypothetical_lens_damage_is_one_service_need_not_product_fact() -> None:
    outcome = _plan("如果镜头裂了可以退吗？")

    assert outcome.early_response is None
    assert [(task.task_type, task.query_target) for task in outcome.tasks] == [
        ("service", "seller_rule.general")
    ]
    assert outcome.tasks[0].metadata["intent_context"]["intent"] == "after_sale.consult"
    assert outcome.tasks[0].metadata["intent_context"]["subject"] == "镜头"
    assert outcome.tasks[0].metadata["intent_context"]["conditions"][0]["modality"] == "hypothetical"
    assert "假设镜头裂了" in outcome.tasks[0].metadata["normalized_question"]


def test_reported_received_lens_damage_is_not_rewritten_as_hypothetical() -> None:
    outcome = _plan("今天早上收到货后发现镜头开裂，可以退吗？")

    assert [(task.task_type, task.query_target) for task in outcome.tasks] == [
        ("service", "seller_rule.general")
    ]
    task = outcome.tasks[0]
    assert task.query == "今天早上收到货后发现镜头开裂，可以退吗"
    assert "买家反馈收到商品时镜头裂了" in task.metadata["normalized_question"]
    assert "假设" not in task.metadata["normalized_question"]
    condition = task.metadata["intent_context"]["conditions"][0]
    assert condition["modality"] == "reported_unverified"
    assert condition["event"] == "镜头裂了"


def test_product_lens_question_and_after_sale_question_stay_separate() -> None:
    outcome = _plan("带什么镜头？如果镜头裂了能退吗？")

    assert [(task.task_type, task.query_target) for task in outcome.tasks] == [
        ("product", "lens.details"),
        ("service", "seller_rule.general"),
    ]
    assert [
        task.metadata["intent_context"]["intent"] for task in outcome.tasks
    ] == ["product.lens_details", "after_sale.consult"]


def test_model_knowledge_question_stays_single_answer_scope() -> None:
    outcome = _plan("这个相机可以测光吗？")

    assert [(task.task_type, task.query_target) for task in outcome.tasks] == [
        ("product", "product.model_knowledge")
    ]
    assert outcome.tasks[0].query == "这个相机可以测光吗"
    assert outcome.tasks[0].metadata["intent_context"]["intent"] == "product.model_knowledge"


def test_vague_followup_without_history_returns_clarification() -> None:
    outcome = asyncio.run(
        _planner().plan_async("那怎么办？", SessionContext(), item_id=None)
    )

    assert outcome.tasks == []
    assert outcome.early_response is not None
    assert outcome.early_response["action"] == "clarify"
    assert outcome.early_response["reason"] == "intent_clarification_required"


def test_same_service_expert_keeps_distinct_after_sale_needs() -> None:
    outcome = _plan("镜头裂了能退吗？退回去运费谁出？")

    assert [task.task_type for task in outcome.tasks] == ["service", "service"]
    assert [
        task.metadata["intent_context"]["intent"] for task in outcome.tasks
    ] == ["after_sale.consult", "after_sale.return_shipping_fee"]


def test_damage_background_attaches_to_return_shipping_fee_need_only() -> None:
    outcome = _plan("今天早上收到货后发现镜头开裂，退回去运费谁出？")

    assert [(task.task_type, task.query_target) for task in outcome.tasks] == [
        ("service", "seller_rule.general")
    ]
    task = outcome.tasks[0]
    assert task.metadata["intent_context"]["intent"] == "after_sale.return_shipping_fee"
    assert task.metadata["normalized_question"] == (
        "买家反馈收到商品时镜头裂了，在未核实前咨询退回商品时运费由谁承担"
    )
    condition = task.metadata["intent_context"]["conditions"][0]
    assert condition == {
        "type": "scenario",
        "event": "镜头裂了",
        "timing": "收到商品时",
        "modality": "reported_unverified",
    }


def test_reported_lens_damage_return_shipping_fee_without_received_time_keeps_condition() -> None:
    outcome = _plan("镜头裂了，寄回去的钱算谁的？")

    assert [(task.task_type, task.query_target) for task in outcome.tasks] == [
        ("service", "seller_rule.general")
    ]
    task = outcome.tasks[0]
    assert task.metadata["intent_context"]["intent"] == "after_sale.return_shipping_fee"
    assert task.metadata["normalized_question"] == (
        "买家反馈当前镜头裂了，在未核实前咨询退回商品时运费由谁承担"
    )
    condition = task.metadata["intent_context"]["conditions"][0]
    assert condition == {
        "type": "scenario",
        "event": "镜头裂了",
        "timing": "unspecified",
        "modality": "reported_unverified",
    }


def test_reported_lens_damage_return_policy_and_shipping_fee_are_two_real_needs() -> None:
    outcome = _plan("今天收到后发现镜头开裂，可以退吗？退回去运费谁出？")

    assert [task.task_type for task in outcome.tasks] == ["service", "service"]
    assert [
        task.metadata["intent_context"]["intent"] for task in outcome.tasks
    ] == ["after_sale.consult", "after_sale.return_shipping_fee"]
    assert [task.query for task in outcome.tasks] == [
        "今天收到后发现镜头开裂，可以退吗",
        "退回去运费谁出",
    ]
    assert [
        task.metadata["normalized_question"] for task in outcome.tasks
    ] == [
        "买家反馈收到商品时镜头裂了，在未核实前确认当前支持的售后处理方式",
        "买家反馈收到商品时镜头裂了，在未核实前咨询退回商品时运费由谁承担",
    ]
    for task in outcome.tasks:
        condition = task.metadata["intent_context"]["conditions"][0]
        assert condition == {
            "type": "scenario",
            "event": "镜头裂了",
            "timing": "收到商品时",
            "modality": "reported_unverified",
        }


def test_no_punctuation_multi_intent_is_not_collapsed() -> None:
    outcome = _plan("这台修过没最低多少今天能发收到有问题咋办")

    assert [task.task_type for task in outcome.tasks] == [
        "product",
        "price",
        "service",
        "service",
    ]
    assert [task.query_target for task in outcome.tasks] == [
        "history.repair_history",
        "price.minimum",
        "shipping.dispatch_time",
        "seller_rule.general",
    ]


def test_conditional_trade_uses_internal_dependency_without_exposing_it() -> None:
    outcome = _plan("如果没修过，1900我就买。")

    assert [(task.task_type, task.query_target) for task in outcome.tasks] == [
        ("product", "history.repair_history"),
        ("price", "price.offer"),
    ]
    assert outcome.tasks[0].metadata["reply_required"] is False
    assert outcome.tasks[1].depends_on_task_ids == ("q1",)

    merged = ResultMerger().merge(
        "如果没修过，1900我就买。",
        outcome.tasks,
        [
            TaskResult("q1", "answered", "没有维修记录。"),
            TaskResult("q2", "answered", "1900 可以。"),
        ],
    )

    assert merged["answer"] == "1900 可以。"


def test_same_price_expert_keeps_multiple_distinct_price_needs() -> None:
    outcome = _plan("最低多少？1900行不行？还能再少吗？")

    assert [task.task_type for task in outcome.tasks] == ["price", "price", "price"]
    assert [task.query_target for task in outcome.tasks] == [
        "price.minimum",
        "price.offer",
        "price.additional_discount",
    ]
