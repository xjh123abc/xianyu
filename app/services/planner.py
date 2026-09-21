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
    xianyu_context_updates,
)
from app.services.xianyu.experts.contracts import ExpertTask


_PLAN_STATE_KEY = "_planner_state"
_GREETING = re.compile(r"(?:你好|您好|哈喽|hello|hi)[！!。？? ]*", re.IGNORECASE)


def _clauses(query: str) -> list[str]:
    """Split explicit buyer questions without assigning domain meaning here."""

    return [part.strip() for part in re.split(r"[。！？?!；;\r\n]+", query) if part.strip()] or [query]


def _unique_tasks(
    expert_tasks: list[ExpertTask],
    metadata: dict[str, object],
) -> list[Task]:
    """Preserve every distinct planned need and remap its dependencies once."""

    selected: list[tuple[str, ExpertTask]] = []
    seen: set[tuple[str, str, str]] = set()
    task_ids: dict[tuple[str, str], str] = {}
    for expert_task in expert_tasks:
        identity = (
            expert_task.expert,
            expert_task.original_question,
            expert_task.query_target,
        )
        if identity in seen:
            continue
        seen.add(identity)
        task_id = f"q{len(selected) + 1}"
        selected.append((task_id, expert_task))
        task_ids[(expert_task.original_question, expert_task.task_id)] = task_id

    tasks: list[Task] = []
    for task_id, expert_task in selected:
        dependencies = [
            task_ids.get(
                (expert_task.original_question, dependency),
                dependency,
            )
            for dependency in expert_task.depends_on_task_ids
        ]
        tasks.append(
            Task(
                task_id,
                expert_task.expert,
                expert_task.original_question,
                {
                    **metadata,
                    "normalized_question": expert_task.normalized_question,
                    "knowledge_scope": expert_task.knowledge_scope,
                    "transaction_conditions": dict(
                        expert_task.transaction_conditions
                    ),
                },
                query_target=expert_task.query_target,
                depends_on_task_ids=tuple(dependencies),
                execution_mode="xianyu_expert",
            )
        )
    return tasks


@dataclass(frozen=True, slots=True)
class PlannerState:
    """Internal transition data carried by every task until S4 owns execution."""

    intent_match: IntentMatch
    question_plan: QuestionPlan
    route: Route
    order_id: str | None
    needs_item: bool
    requires_item_resolution: bool
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
            requires_item_resolution=needs_item and route == "rag",
            defer_to_order_context=remembered_order_id is not None,
            is_rule_followup=is_rule_followup(query, remembered_order_id),
            use_xianyu_without_item=(
                context.current_item_id is not None
                or is_seller_scoped_query(query)
                or bool(_GREETING.fullmatch(query))
            ),
        )
        metadata = {_PLAN_STATE_KEY: _state_metadata(state)}

        if route in {"order", "missing_order_id"}:
            return [
                Task(
                    "order-1",
                    "order",
                    query,
                    metadata,
                    execution_mode=(
                        "missing_order_id" if route == "missing_order_id" else "order"
                    ),
                )
            ]
        if route == "rag_mcp":
            # The old combined route is classified only here.  Its execution
            # is two ordinary tasks, never OrderChatHandler.combined().
            return [
                Task("order-1", "order", query, metadata, execution_mode="order"),
                Task(
                    "service-1",
                    "service",
                    query,
                    metadata,
                    execution_mode="common_knowledge",
                ),
            ]
        if route == "unsupported_action":
            return [
                Task(
                    "service-1",
                    "service",
                    query,
                    metadata,
                    execution_mode="unsupported_action",
                )
            ]

        execution_mode = (
            "general_rag"
            if state.is_rule_followup
            else (
                "xianyu_expert"
                if state.needs_item or state.use_xianyu_without_item
                else "general_rag"
            )
        )
        if execution_mode == "general_rag":
            # Ordinary RAG receives the complete buyer message.  Clause
            # splitting is only for the expert Task[] classification path.
            return [
                Task(
                    "service-1",
                    "service",
                    query,
                    metadata,
                    execution_mode="general_rag",
                )
            ]

        xianyu_context = dict(context.platform_context.get("xianyu", {}))
        effective_item_id = item_id or context.current_item_id
        if effective_item_id is not None and not xianyu_context.get("item_id"):
            xianyu_context["item_id"] = effective_item_id
        clauses = _clauses(query)
        clause_plans = [
            (
                clause,
                build_expert_plan(
                    clause,
                    intent_router=self._intent_router,
                    history=context.history,
                    xianyu_context=xianyu_context,
                ),
            )
            for clause in clauses
        ]
        planned_expert_tasks: list[ExpertTask] = []
        for index, (clause, expert_tasks) in enumerate(clause_plans, start=1):
            if expert_tasks:
                planned_expert_tasks.extend(expert_tasks)
                continue
            # The former orchestrator fallback is now Planner-owned so the
            # execution path never needs to classify this clause again.
            clause_question_plan = build_question_plan(clause)
            clause_needs_item = self._requires_item_context(clause) or any(
                question["scope"] == "item"
                for question in clause_question_plan["knowledge_questions"]
            )
            planned_expert_tasks.append(
                ExpertTask(
                    task_id=f"fallback-{index}",
                    expert="product" if clause_needs_item else "service",
                    question_fragment=clause,
                    normalized_question=(
                        "商品专项知识" if clause_needs_item else "卖家通用规则"
                    ),
                    knowledge_scope=(
                        "model_knowledge" if clause_needs_item else "seller_rule"
                    ),
                    query_target=(
                        "product.model_knowledge"
                        if clause_needs_item
                        else "seller_rule.general"
                    ),
                    original_question=clause,
                )
            )
        return _unique_tasks(planned_expert_tasks, metadata)

    @staticmethod
    def xianyu_context_updates(
        query: str,
        context: Mapping[str, object] | None = None,
    ) -> dict[str, str]:
        """Expose planner-owned follow-up state extraction through one boundary."""

        return xianyu_context_updates(query, context)

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
        and isinstance(
            raw_state.get(
                "requires_item_resolution",
                raw_state.get("needs_item") and route == "rag",
            ),
            bool,
        )
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
        requires_item_resolution=raw_state.get(
            "requires_item_resolution",
            raw_state["needs_item"] and route == "rag",
        ),
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
        "requires_item_resolution": state.requires_item_resolution,
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
