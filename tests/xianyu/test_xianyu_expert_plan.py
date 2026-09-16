"""S4 planning and product-scoped follow-up state checks."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from app.services.intent_router import IntentRouter
from app.services.query_planner import build_expert_plan
from app.services.session_manager import SessionManager


class PlannerSpy:
    def __init__(self, payload: Mapping[str, object]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, object]] = []

    def __call__(self, query: str, *, history: object = None) -> Mapping[str, object]:
        self.calls.append((query, history))
        return self.payload


def _by_expert(tasks, expert: str):
    return [task for task in tasks if task.expert == expert]


def test_planning_precheck_never_uses_intent_classifier() -> None:
    classifier_calls: list[str] = []
    router = IntentRouter(classifier=lambda query: classifier_calls.append(query) or "PRICE")

    tasks = build_expert_plan("还在吗有没有维修过不包邮最低多少", intent_router=router)

    assert classifier_calls == []
    assert [task.expert for task in tasks] == ["product", "product", "price"]
    assert [task.normalized_question for task in tasks] == ["是否在售", "是否维修过", "最低价"]


def test_simple_question_uses_rule_plan_without_model_call() -> None:
    planner = PlannerSpy({"tasks": []})

    tasks = build_expert_plan("这个还在吗？", planner=planner)

    assert planner.calls == []
    assert [(task.expert, task.normalized_question) for task in tasks] == [
        ("product", "是否在售")
    ]


@pytest.mark.parametrize(
    ("query", "expert", "normalized_question"),
    [
        ("现在还能拍吗？", "product", "是否在售"),
        ("东西卖掉了吗？", "product", "是否在售"),
        ("现在下单还有货吗？", "product", "是否在售"),
        ("现在什么价出？", "price", "商品标价"),
    ],
)
def test_batch_phrasing_uses_deterministic_fact_plan(
    query: str,
    expert: str,
    normalized_question: str,
) -> None:
    planner = PlannerSpy({"tasks": []})

    tasks = build_expert_plan(query, planner=planner)

    assert planner.calls == []
    assert [(task.expert, task.normalized_question) for task in tasks] == [
        (expert, normalized_question)
    ]


def test_complex_no_punctuation_keeps_all_rule_tasks_when_model_plan_is_empty() -> None:
    planner = PlannerSpy({"tasks": []})

    tasks = build_expert_plan("还在吗有没有维修过不包邮最低多少", planner=planner)

    assert len(planner.calls) == 1
    assert [task.expert for task in tasks] == ["product", "product", "price"]
    assert tasks[-1].transaction_conditions == {
        "shipping": "buyer_pays",
        "request_kind": "minimum",
    }


def test_planner_failure_keeps_the_deterministic_task_baseline() -> None:
    def unavailable_planner(query: str, *, history: object = None) -> Mapping[str, object]:
        raise RuntimeError("planner unavailable")

    tasks = build_expert_plan(
        "还在吗有没有维修过不包邮最低多少",
        planner=unavailable_planner,
    )

    assert [task.expert for task in tasks] == ["product", "product", "price"]


def test_conditional_offer_preserves_shipping_and_depends_on_repair() -> None:
    planner = PlannerSpy({"tasks": []})

    tasks = build_expert_plan("如果没修过，1470不包邮我就买", planner=planner)

    assert len(planner.calls) == 1
    repair = next(task for task in tasks if task.normalized_question == "是否维修过")
    price = next(task for task in tasks if task.expert == "price")
    assert price.transaction_conditions == {
        "shipping": "buyer_pays",
        "request_kind": "offer",
        "offer_cents": 147000,
    }
    assert price.depends_on_task_ids == (repair.task_id,)


def test_planner_cannot_invent_price_or_replace_missing_rule_task() -> None:
    planner = PlannerSpy(
        {
            "tasks": [
                {
                    "task_id": "made_up_price",
                    "expert": "price",
                    "question_fragment": "不包邮最低多少",
                    "normalized_question": "不包邮最低价",
                    "knowledge_scope": "item_fact",
                    "transaction_conditions": {
                        "shipping": "buyer_pays",
                        "request_kind": "offer",
                        "offer_cents": 999999,
                    },
                    "depends_on_task_ids": [],
                }
            ]
        }
    )

    tasks = build_expert_plan("还在吗？修过没有？不包邮最低多少？", planner=planner)

    assert len(planner.calls) == 1
    assert [task.expert for task in tasks] == ["product", "product", "price"]
    price = tasks[-1]
    assert price.transaction_conditions == {
        "shipping": "buyer_pays",
        "request_kind": "minimum",
    }


def test_conditional_model_price_without_repair_dependency_falls_back_to_rule_plan() -> None:
    planner = PlannerSpy(
        {
            "tasks": [
                {
                    "task_id": "repair",
                    "expert": "product",
                    "question_fragment": "修过",
                    "normalized_question": "是否维修过",
                    "knowledge_scope": "item_fact",
                    "transaction_conditions": {},
                    "depends_on_task_ids": [],
                },
                {
                    "task_id": "price",
                    "expert": "price",
                    "question_fragment": "1470不包邮我就买",
                    "normalized_question": "买家报价",
                    "knowledge_scope": "item_fact",
                    "transaction_conditions": {
                        "shipping": "buyer_pays",
                        "request_kind": "offer",
                        "offer_cents": 147000,
                    },
                    "depends_on_task_ids": [],
                },
            ]
        }
    )

    tasks = build_expert_plan("如果没修过，1470不包邮我就买", planner=planner)

    repair = next(task for task in tasks if task.normalized_question == "是否维修过")
    price = next(task for task in tasks if task.expert == "price")
    assert price.depends_on_task_ids == (repair.task_id,)


def test_follow_up_uses_context_topic_but_never_treats_it_as_price_authority() -> None:
    planner = PlannerSpy({"tasks": []})

    tasks = build_expert_plan(
        "那不包邮呢？",
        planner=planner,
        xianyu_context={"recent_price_topic": "minimum"},
    )

    assert len(planner.calls) == 1
    assert len(tasks) == 1
    assert tasks[0].transaction_conditions == {
        "shipping": "buyer_pays",
        "request_kind": "minimum",
    }


def test_shipping_comparison_is_not_a_selected_shipping_condition() -> None:
    tasks = build_expert_plan("包邮最低多少？不包邮呢？")

    assert len(tasks) == 1
    assert tasks[0].transaction_conditions == {
        "shipping_comparison": True,
        "request_kind": "minimum",
    }


def test_session_context_defaults_and_isolation_preserve_existing_order_state() -> None:
    manager = SessionManager()
    manager.append_turn("first", "查 TEST1001", "已找到", order_id="TEST1001")
    manager.update_xianyu_context(
        "first",
        item_id="canon_ftb_001",
        recent_price_topic="minimum",
        shipping_condition="buyer_pays",
    )

    first_history, first_state = SessionManager.read_context(manager.get_or_create("first")[1])
    second_history, second_state = SessionManager.read_context(manager.get_or_create("second")[1])

    assert first_history
    assert first_state["order_id"] == "TEST1001"
    assert first_state["xianyu_context"] == {
        "item_id": "CANON_FTB_001",
        "recent_price_topic": "minimum",
        "shipping_condition": "buyer_pays",
    }
    assert second_history == []
    assert second_state["xianyu_context"] == {
        "item_id": None,
        "recent_price_topic": None,
        "shipping_condition": None,
    }


def test_old_session_state_gets_xianyu_defaults_without_changing_order_fields() -> None:
    history, state = SessionManager.read_context(
        {
            "history": [{"role": "user", "content": "查 TEST1001"}],
            "state": {
                "order_id": "TEST1001",
                "current_item_id": "CANON_FTB_001",
                "last_intent": "order_query",
            },
        }
    )

    assert history == [{"role": "user", "content": "查 TEST1001"}]
    assert state["order_id"] == "TEST1001"
    assert state["current_item_id"] == "CANON_FTB_001"
    assert state["last_intent"] == "order_query"
    assert state["xianyu_context"] == {
        "item_id": None,
        "recent_price_topic": None,
        "shipping_condition": None,
    }


def test_switching_item_clears_only_that_chats_transaction_conditions() -> None:
    manager = SessionManager()
    manager.set_current_item_id("first", "ITEM_A")
    manager.update_xianyu_context(
        "first",
        recent_price_topic="minimum",
        shipping_condition="buyer_pays",
    )
    manager.set_current_item_id("second", "ITEM_A")
    manager.update_xianyu_context(
        "second",
        recent_price_topic="minimum",
        shipping_condition="seller_pays",
    )

    manager.set_current_item_id("first", "ITEM_B")

    assert manager.get_xianyu_context("first") == {
        "item_id": "ITEM_B",
        "recent_price_topic": None,
        "shipping_condition": None,
    }
    assert manager.get_xianyu_context("second") == {
        "item_id": "ITEM_A",
        "recent_price_topic": "minimum",
        "shipping_condition": "seller_pays",
    }
