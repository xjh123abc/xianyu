"""Regression coverage for complete, target-owned Xianyu sub-questions."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

from app.services.item_service import ItemService
from app.services.query_planner import build_expert_plan
from app.services.xianyu.experts.contracts import ExpertContext
from app.services.xianyu.experts.product_agent import ProductAgent
from app.services.xianyu.experts.service_agent import ServiceAgent
from app.services.xianyu.item_fact_responder import ItemFactResponder


def _item() -> dict[str, object]:
    return ItemService().get_item_info("CANON_FTB_001")


def test_three_delimited_questions_keep_complete_source_and_distinct_targets() -> None:
    tasks = build_expert_plan("走什么快递 多久能发货 是50mm镜头吗")

    assert [task.original_question for task in tasks] == [
        "走什么快递",
        "多久能发货",
        "是50mm镜头吗",
    ]
    assert [task.query_target for task in tasks] == [
        "shipping.carrier",
        "shipping.dispatch_time",
        "lens.focal_length_mm",
    ]


def test_compact_multi_question_never_becomes_keyword_slices() -> None:
    query = "走什么快递多久能发货是50mm镜头吗"

    tasks = build_expert_plan(query)

    assert len(tasks) == 3
    assert all(task.original_question == query for task in tasks)
    assert [task.query_target for task in tasks] == [
        "shipping.carrier",
        "shipping.dispatch_time",
        "lens.focal_length_mm",
    ]


def test_four_or_more_delimited_questions_each_keep_a_distinct_target() -> None:
    tasks = build_expert_plan("走什么快递 多久能发货 是50mm镜头吗 包邮吗")

    assert [task.query_target for task in tasks] == [
        "shipping.carrier",
        "shipping.dispatch_time",
        "lens.focal_length_mm",
        "shipping.fee",
    ]
    assert [task.original_question for task in tasks] == [
        "走什么快递",
        "多久能发货",
        "是50mm镜头吗",
        "包邮吗",
    ]


def test_experts_use_the_planned_target_without_reclassifying_the_question() -> None:
    query = "走什么快递 多久能发货 是50mm镜头吗"
    tasks = build_expert_plan(query)
    item = _item()
    responder = ItemFactResponder()
    context = ExpertContext(query=query, item=item)
    service = ServiceAgent(
        fact_responder=responder,
        prepare_evidence=AsyncMock(),
        generator=Mock(),
    )
    product = ProductAgent(
        fact_responder=responder,
        prepare_evidence=AsyncMock(),
        generator=Mock(),
    )

    service_results = asyncio.run(
        service.run([task for task in tasks if task.expert == "service"], context)
    )
    product_results = asyncio.run(
        product.run([task for task in tasks if task.expert == "product"], context)
    )

    assert [result.answer for result in service_results] == [
        "中通。",
        "付款后 48 小时内发出。",
    ]
    assert product_results[0].answer is not None
    assert "50mm 焦段" in product_results[0].answer


def test_validated_model_question_replaces_only_the_rule_question_for_its_target() -> None:
    def planner(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "tasks": [
                {
                    "task_id": "carrier",
                    "expert": "service",
                    "original_question": "走什么快递",
                    "normalized_question": "使用哪家快递",
                    "query_target": "shipping.carrier",
                    "knowledge_scope": "item_fact",
                    "transaction_conditions": {},
                    "depends_on_task_ids": [],
                }
            ]
        }

    tasks = build_expert_plan(
        "走什么快递多久能发货",
        planner=planner,
        xianyu_context={"item_id": "CANON_FTB_001"},
    )

    carrier = next(task for task in tasks if task.query_target == "shipping.carrier")
    dispatch = next(task for task in tasks if task.query_target == "shipping.dispatch_time")
    assert carrier.original_question == "走什么快递"
    assert dispatch.original_question == "走什么快递多久能发货"


def test_model_keyword_slice_cannot_replace_a_complete_rule_question() -> None:
    def planner(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "tasks": [
                {
                    "task_id": "carrier",
                    "expert": "service",
                    "original_question": "快递 多久能发货",
                    "normalized_question": "使用哪家快递",
                    "query_target": "shipping.carrier",
                    "knowledge_scope": "item_fact",
                    "transaction_conditions": {},
                    "depends_on_task_ids": [],
                }
            ]
        }

    query = "走什么快递 多久能发货"
    tasks = build_expert_plan(
        query,
        planner=planner,
        xianyu_context={"item_id": "CANON_FTB_001"},
    )

    carrier = next(task for task in tasks if task.query_target == "shipping.carrier")
    assert carrier.original_question == "走什么快递"


def test_model_price_target_must_agree_with_its_validated_conditions() -> None:
    def planner(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "tasks": [
                {
                    "task_id": "price",
                    "expert": "price",
                    "original_question": "不包邮最低多少",
                    "normalized_question": "不包邮最低价",
                    "query_target": "price.listed_price",
                    "knowledge_scope": "item_fact",
                    "transaction_conditions": {
                        "shipping": "buyer_pays",
                        "request_kind": "minimum",
                    },
                    "depends_on_task_ids": [],
                }
            ]
        }

    tasks = build_expert_plan("不包邮最低多少", planner=planner)

    assert [task.query_target for task in tasks] == ["price.minimum"]


def test_quality_problem_keeps_common_rule_when_model_only_plans_price() -> None:
    """Deterministic rules must retain every known need from a compound turn."""

    def planner(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "tasks": [
                {
                    "task_id": "price",
                    "expert": "price",
                    "original_question": "多少钱",
                    "normalized_question": "商品标价",
                    "query_target": "price.listed_price",
                    "knowledge_scope": "item_fact",
                    "transaction_conditions": {"request_kind": "listed_price"},
                    "depends_on_task_ids": [],
                }
            ]
        }

    tasks = build_expert_plan(
        "多少钱？收到有质量问题怎么办？",
        planner=planner,
        xianyu_context={"item_id": "DEMO_ITEM_001"},
    )

    assert [task.query_target for task in tasks] == [
        "price.listed_price",
        "seller_rule.general",
    ]
    assert tasks[1].knowledge_scope == "seller_rule"


def test_after_sale_follow_up_uses_common_rule_with_current_item() -> None:
    tasks = build_expert_plan(
        "那售后怎么处理？",
        xianyu_context={"item_id": "DEMO_ITEM_001"},
    )

    assert len(tasks) == 1
    assert tasks[0].expert == "service"
    assert tasks[0].knowledge_scope == "seller_rule"
    assert tasks[0].query_target == "seller_rule.general"
