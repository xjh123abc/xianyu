"""One task-planning entry point built on the existing routing modules.

This is deliberately an adapter, not a replacement for the mature routers.
The private compatibility state lets the pre-S4 ChatService preserve its
existing executors while it consumes one public ``Task`` list.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from app.services.chat_contracts import SessionContext, Task
from app.services.intent_router import IntentMatch, IntentRouter
from app.services.order_router import (
    Route,
    history_order_id,
    is_rule_followup,
    route_query,
    session_order_id,
)
from app.services.query_planner import (
    QuestionPlan,
    build_expert_plan,
    build_question_plan,
    is_seller_scoped_query,
)


_PLAN_STATE_KEY = "_planner_state"
_GREETING = re.compile(r"(?:你好|您好|哈喽|hello|hi)[！!。？? ]*", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class PlannerState:
    """Internal transition data carried by every task until S4 owns execution."""

    intent_match: IntentMatch
    question_plan: QuestionPlan
    route: Route
    order_id: str | None
    needs_item: bool
    defer_to_order_context: bool
    is_rule_followup: bool
    use_xianyu_without_item: bool


class Planner:
    """Return unified tasks while reusing the proven rule-first classifiers."""

    def __init__(
        self,
        *,
        intent_router: IntentRouter,
        requires_item_context: Callable[[str], bool],
        may_contain_explicit_item_reference: Callable[[str], bool],
    ) -> None:
        self._intent_router = intent_router
        self._requires_item_context = requires_item_context
        self._may_contain_explicit_item_reference = may_contain_explicit_item_reference

    def plan(
        self,
        message: str,
        context: SessionContext,
        *,
        item_id: str | None = None,
    ) -> list[Task]:
        """Classify one turn once and expose its work as ``Task`` values only."""

        query = str(message or "").strip()
        question_plan = build_question_plan(query)
        intent_match = self._intent_router.route(query, allow_ai=False)
        remembered_order_id = session_order_id(
            {"order_id": context.current_order_id}
        ) or history_order_id(context.history)
        route, order_id = route_query(
            query,
            context.history,
            {"order_id": context.current_order_id},
        )
        needs_item = self._needs_item(
            query,
            context,
            item_id,
            intent_match,
            question_plan,
        )
        state = PlannerState(
            intent_match=intent_match,
            question_plan=question_plan,
            route=route,
            order_id=order_id,
            needs_item=needs_item,
            defer_to_order_context=remembered_order_id is not None,
            is_rule_followup=is_rule_followup(query, remembered_order_id),
            use_xianyu_without_item=(
                context.current_item_id is not None
                or is_seller_scoped_query(query)
                or bool(_GREETING.fullmatch(query))
            ),
        )
        metadata = {_PLAN_STATE_KEY: _state_metadata(state)}

        if route in {"order", "rag_mcp", "missing_order_id"}:
            return [Task("order-1", "order", query, metadata)]
        if route == "unsupported_action":
            return [Task("service-1", "service", query, metadata)]

        xianyu_context = context.platform_context.get("xianyu", {})
        tasks = [
            Task(expert_task.task_id, expert_task.expert, expert_task.original_question, metadata)
            for expert_task in build_expert_plan(
                query,
                intent_router=self._intent_router,
                history=context.history,
                xianyu_context=xianyu_context,
            )
        ]
        # Ordinary RAG has no expert task today.  It remains one service task
        # so Planner has a total, Task[]-only contract before S4's executor.
        return tasks or [Task("service-1", "service", query, metadata)]

    def _needs_item(
        self,
        query: str,
        context: SessionContext,
        item_id: str | None,
        intent_match: IntentMatch,
        question_plan: QuestionPlan,
    ) -> bool:
        return bool(
            item_id
            or context.current_item_id
            or question_plan["item_fields"]
            or any(
                question["scope"] == "item"
                for question in question_plan["knowledge_questions"]
            )
            or (
                intent_match.requires_item
                and bool(
                    item_id
                    or context.current_item_id
                    or self._requires_item_context(query)
                )
            )
            or self._may_contain_explicit_item_reference(query)
        )


def planner_state(tasks: list[Task]) -> PlannerState:
    """Read transition-only execution data without exposing a second result type."""

    if not tasks:
        raise ValueError("Planner must return at least one Task")
    raw_state = tasks[0].metadata.get(_PLAN_STATE_KEY)
    if not isinstance(raw_state, Mapping):
        raise ValueError("Planner tasks are missing planner state")
    raw_intent = raw_state.get("intent_match")
    raw_plan = raw_state.get("question_plan")
    if not isinstance(raw_intent, Mapping) or not isinstance(raw_plan, Mapping):
        raise ValueError("Planner tasks contain invalid planner state")
    intent = raw_intent.get("intent")
    fields = raw_intent.get("required_fields")
    source = raw_intent.get("source")
    item_fields = raw_plan.get("item_fields")
    knowledge_questions = raw_plan.get("knowledge_questions")
    route = raw_state.get("route")
    if not (
        isinstance(intent, str)
        and isinstance(fields, list)
        and all(isinstance(field, str) for field in fields)
        and source in {"rule", "ai", "fallback"}
        and isinstance(item_fields, list)
        and isinstance(knowledge_questions, list)
        and all(isinstance(question, Mapping) for question in knowledge_questions)
        and route in {"rag", "order", "rag_mcp", "missing_order_id", "unsupported_action"}
        and isinstance(raw_state.get("needs_item"), bool)
        and isinstance(raw_state.get("defer_to_order_context"), bool)
        and isinstance(raw_state.get("is_rule_followup"), bool)
        and isinstance(raw_state.get("use_xianyu_without_item"), bool)
    ):
        raise ValueError("Planner tasks contain invalid planner state")
    return PlannerState(
        intent_match=IntentMatch(intent, tuple(fields), source),  # type: ignore[arg-type]
        question_plan={
            "item_fields": list(item_fields),
            "knowledge_questions": [dict(question) for question in knowledge_questions],
        },
        route=route,
        order_id=_optional_order_id(raw_state.get("order_id")),
        needs_item=raw_state["needs_item"],
        defer_to_order_context=raw_state["defer_to_order_context"],
        is_rule_followup=raw_state["is_rule_followup"],
        use_xianyu_without_item=raw_state["use_xianyu_without_item"],
    )


def _state_metadata(state: PlannerState) -> dict[str, object]:
    """Keep Task metadata transport-safe during the S3-to-S4 transition."""

    return {
        "intent_match": {
            "intent": state.intent_match.intent,
            "required_fields": list(state.intent_match.required_fields),
            "source": state.intent_match.source,
        },
        "question_plan": {
            "item_fields": list(state.question_plan["item_fields"]),
            "knowledge_questions": [
                dict(question) for question in state.question_plan["knowledge_questions"]
            ],
        },
        "route": state.route,
        "order_id": state.order_id,
        "needs_item": state.needs_item,
        "defer_to_order_context": state.defer_to_order_context,
        "is_rule_followup": state.is_rule_followup,
        "use_xianyu_without_item": state.use_xianyu_without_item,
    }


def _optional_order_id(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Planner tasks contain invalid order_id")
    return value
