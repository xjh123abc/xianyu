"""S3 acceptance tests for the unified Task[] planning boundary."""

from __future__ import annotations

import inspect

import app.services.planner as planner_module
from app.services.chat_contracts import SessionContext
from app.services.chat_service import ChatService
from app.services.intent_router import IntentRouter
from app.services.planner import Planner, planner_state
from app.services.task_executor import XianyuExpertTaskHandler
from app.services.xianyu.item_context_resolver import ItemContextResolver


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


def test_planner_preserves_the_complete_expert_execution_contract() -> None:
    tasks = _planner().plan(
        "这个修过吗？最低多少？多久发货？",
        SessionContext(current_item_id="ITEM-001"),
    )

    assert [task.query_target for task in tasks] == [
        "history.repair_history",
        "price.minimum",
        "shipping.dispatch_time",
    ]
    assert all(task.execution_mode == "xianyu_expert" for task in tasks)
    assert all(isinstance(task.depends_on_task_ids, tuple) for task in tasks)
    assert all("normalized_question" in task.metadata for task in tasks)
    assert all("knowledge_scope" in task.metadata for task in tasks)
    assert all("query_target" not in task.metadata for task in tasks)
    assert all("depends_on_task_ids" not in task.metadata for task in tasks)
    assert all("execution_mode" not in task.metadata for task in tasks)


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


def test_planner_builds_each_clause_once_before_deduplicating_tasks(monkeypatch) -> None:
    calls: list[str] = []

    def _empty_plan(query: str, **kwargs: object) -> list[object]:
        del kwargs
        calls.append(query)
        return []

    monkeypatch.setattr(planner_module, "build_expert_plan", _empty_plan)

    _planner().plan("这个修过吗？最低多少？", SessionContext(current_item_id="ITEM-001"))

    assert calls == ["这个修过吗", "最低多少"]


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


def test_chat_service_does_not_keep_old_route_or_knowledge_execution_branches() -> None:
    source = inspect.getsource(ChatService._execute_planned_turn)

    for legacy_execution in (
        "state.route",
        "handle_common(",
        "self.chat(",
        "order_handler.order(",
        "common_knowledge_query(",
    ):
        assert legacy_execution not in source


def test_item_resolver_does_not_repeat_planner_order_routing() -> None:
    source = inspect.getsource(ItemContextResolver.resolve)

    assert "route_query(" not in source


def test_main_expert_handler_executes_preplanned_tasks_without_replanning() -> None:
    source = inspect.getsource(XianyuExpertTaskHandler.handle)

    assert "execute_tasks(" in source
    assert "expert_orchestrator.handle(" not in source
    assert "build_expert_plan(" not in source
