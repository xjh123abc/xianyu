"""S3 acceptance tests for the unified Task[] planning boundary."""

from __future__ import annotations

import inspect

from app.services.chat_contracts import SessionContext
from app.services.chat_service import ChatService
from app.services.intent_router import IntentRouter
from app.services.planner import Planner, planner_state


def _planner() -> Planner:
    return Planner(
        intent_router=IntentRouter(),
        requires_item_context=lambda query: any(
            term in query
            for term in (
                "\u8fd9\u4e2a\u5546\u54c1",
                "\u8fd9\u4e2a",
                "\u4fee\u8fc7",
                "\u4ef7\u683c",
                "\u53d1\u8d27",
            )
        ),
        may_contain_explicit_item_reference=lambda query: "ITEM-" in query,
    )


def test_planner_reuses_legacy_rules_to_create_one_task_per_buyer_need() -> None:
    tasks = _planner().plan(
        "\u8fd9\u4e2a\u4fee\u8fc7\u5417\uff1f\u6700\u4f4e\u591a\u5c11\uff1f\u591a\u4e45\u53d1\u8d27\uff1f",
        SessionContext(
            current_item_id="ITEM-001",
            platform_context={"xianyu": {"item_id": "ITEM-001"}},
        ),
    )

    assert [task.task_type for task in tasks] == ["product", "price", "service"]
    assert planner_state(tasks).route == "rag"
    assert planner_state(tasks).needs_item is True


def test_planner_classifies_each_delimited_clause_with_the_single_question_rules() -> None:
    context = SessionContext(current_item_id="ITEM-001")

    assert [task.task_type for task in _planner().plan("周日能到吗？", context)] == ["service"]
    assert [task.task_type for task in _planner().plan(
        "这个相机修过吗？最低多少？周日能到吗？", context
    )] == ["product", "price", "service"]
    assert [task.task_type for task in _planner().plan(
        "这个修过吗？最低多少？多久发货？", context
    )] == ["product", "price", "service"]


def test_planner_keeps_the_existing_order_route_as_one_order_task() -> None:
    tasks = _planner().plan("TEST1001 \u5230\u54ea\u4e86\uff1f", SessionContext())

    assert [task.task_type for task in tasks] == ["order"]
    assert planner_state(tasks).route == "order"
    assert planner_state(tasks).order_id == "TEST1001"


def test_chat_service_uses_the_planner_boundary_instead_of_direct_router_calls() -> None:
    source = inspect.getsource(ChatService._chat_async_locked)

    assert "self.planner.plan(query, context, item_id=item_id)" in source
    for legacy_entrypoint in (
        "self.intent_router.route",
        "build_question_plan(",
        "route_query(",
        "is_rule_followup(",
        "_should_use_xianyu_experts",
    ):
        assert legacy_entrypoint not in source
