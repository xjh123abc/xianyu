"""One task-planning entry point built on the existing routing modules.

This is deliberately an adapter, not a replacement for the mature routers.
The private compatibility state lets the pre-S4 ChatService preserve its
existing executors while it consumes one public ``Task`` list.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from app.services.chat_contracts import SessionContext, Task
from app.services.chat_response import non_rag_response
from app.services.intent_analyzer import IntentAnalyzer
from app.services.intent_contracts import UnderstandingResult, UserNeed
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


@dataclass(frozen=True, slots=True)
class PlanOutcome:
    """Planner result for the async semantic-planning boundary."""

    tasks: list[Task]
    early_response: dict[str, object] | None = None
    understanding: UnderstandingResult | None = None


class Planner:
    """Return unified tasks while reusing the proven rule-first classifiers."""

    def __init__(
        self,
        *,
        intent_router: IntentRouter,
        requires_item_context: Callable[[str], bool],
        may_contain_explicit_item_reference: Callable[[str], bool],
        intent_analyzer: IntentAnalyzer | None = None,
    ) -> None:
        self._intent_router = intent_router
        self._requires_item_context = requires_item_context
        self._may_contain_explicit_item_reference = may_contain_explicit_item_reference
        self._intent_analyzer = intent_analyzer or IntentAnalyzer()

    async def plan_async(
        self,
        message: str,
        context: SessionContext,
        *,
        item_id: str | None = None,
    ) -> PlanOutcome:
        """Understand the whole buyer turn before creating executable tasks."""

        query = str(message or "").strip()
        state = self._build_state(query, context, item_id)
        if _keeps_legacy_boundary(query, state):
            return PlanOutcome(self._tasks_from_state(query, context, item_id, state))

        understanding = await self._intent_analyzer.analyze(
            query,
            context,
            item_id=item_id,
        )
        if understanding.status == "clarify":
            response = non_rag_response(
                query,
                understanding.clarification_question
                or "请补充你想确认的具体问题。",
                can_answer=False,
                route="unified",
                action="clarify",
            )
            response["reason"] = "intent_clarification_required"
            return PlanOutcome([], response, understanding)
        if understanding.status == "error":
            if _can_use_legacy_semantic_fallback(query, state, understanding):
                return PlanOutcome(
                    self._tasks_from_state(query, context, item_id, state),
                    understanding=understanding,
                )
            response = non_rag_response(
                query,
                "暂时没能解析这条消息，请稍后再试或换个说法。",
                can_answer=False,
                route="unified",
                action="reply",
            )
            response["reason"] = understanding.error_reason or "intent_parse_failed"
            return PlanOutcome([], response, understanding)

        tasks = _tasks_from_understanding(
            query,
            context,
            item_id,
            understanding,
        )
        if not tasks:
            response = non_rag_response(
                query,
                "暂时没能解析这条消息，请稍后再试或换个说法。",
                can_answer=False,
                route="unified",
                action="reply",
            )
            response["reason"] = "intent_task_mapping_empty"
            return PlanOutcome([], response, understanding)
        return PlanOutcome(tasks, understanding=understanding)

    def plan(
        self,
        message: str,
        context: SessionContext,
        *,
        item_id: str | None = None,
    ) -> list[Task]:
        """Classify one turn once and expose its work as ``Task`` values only."""

        query = str(message or "").strip()
        state = self._build_state(query, context, item_id)
        return self._tasks_from_state(query, context, item_id, state)

    def _build_state(
        self,
        query: str,
        context: SessionContext,
        item_id: str | None,
    ) -> PlannerState:
        """Classify route and item needs without constructing executable tasks."""

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
        return state

    def _tasks_from_state(
        self,
        query: str,
        context: SessionContext,
        item_id: str | None,
        state: PlannerState,
    ) -> list[Task]:
        """Materialize the compatibility task plan only when it will be used."""

        metadata = {_PLAN_STATE_KEY: _state_metadata(state)}

        if state.route in {"order", "missing_order_id"}:
            return [
                Task(
                    "order-1",
                    "order",
                    query,
                    metadata,
                    execution_mode=(
                        "missing_order_id"
                        if state.route == "missing_order_id"
                        else "order"
                    ),
                )
            ]
        if state.route == "rag_mcp":
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
        if state.route == "unsupported_action":
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
            product_scoped_fallback = clause_needs_item or (
                effective_item_id is not None
                and not is_seller_scoped_query(clause)
                and _looks_like_model_knowledge(clause)
            )
            planned_expert_tasks.append(
                ExpertTask(
                    task_id=f"fallback-{index}",
                    expert="product" if product_scoped_fallback else "service",
                    question_fragment=clause,
                    normalized_question=clause,
                    knowledge_scope=(
                        "model_knowledge" if product_scoped_fallback else "seller_rule"
                    ),
                    query_target=(
                        "product.model_knowledge"
                        if product_scoped_fallback
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


def _uses_xianyu_expert_path(state: PlannerState) -> bool:
    if state.route in {"order", "missing_order_id", "rag_mcp", "unsupported_action"}:
        return False
    if state.is_rule_followup:
        return False
    return state.needs_item or state.use_xianyu_without_item


def _keeps_legacy_boundary(query: str, state: PlannerState) -> bool:
    """Leave non-Xianyu expert paths on the existing stable plan."""

    if query.strip("？?。!！ ") in {"那怎么办", "怎么办", "咋办", "那咋办"}:
        return False
    if _requires_semantic_conversation_boundary(query):
        return False
    return not _uses_xianyu_expert_path(state)


def _requires_semantic_conversation_boundary(query: str) -> bool:
    """Let ordinary chat and no-reply turns bypass legacy general RAG."""

    lowered = str(query or "").casefold()
    cleaned = lowered.strip("，,。.!！?？、；;~ ")
    if re.fullmatch(
        r"(?:谢谢|谢了|感谢|多谢|辛苦了|thanks|thank\s+you|thx)[！!。？?~ ]*",
        cleaned,
        re.IGNORECASE,
    ):
        return True
    if re.fullmatch(
        r"(?:好|好的|好嘞|ok|okay|收到|明白|知道了|嗯|嗯嗯|不用了|先不用了|没事了|先这样)[！!。？?~ ]*",
        cleaned,
        re.IGNORECASE,
    ):
        return True
    return any(
        term in lowered
        for term in ("已读", "已送达", "系统消息", "系统事件", "不用回复", "无需回复")
    )


def _can_use_legacy_semantic_fallback(
    query: str,
    state: PlannerState,
    understanding: UnderstandingResult,
) -> bool:
    """Keep legacy item fallbacks only when they cannot revive after-sale overreach."""

    if understanding.error_reason not in {
        "intent_semantic_model_unavailable",
        "intent_model_failed",
        "intent_model_timeout",
    }:
        return False
    if not _uses_xianyu_expert_path(state):
        return False
    lowered = query.casefold()
    risky_terms = (
        "退",
        "退款",
        "售后",
        "裂",
        "裂痕",
        "质量问题",
        "有问题",
        "坏",
        "怎么办",
        "咋办",
        "无理由",
    )
    return not any(term in lowered for term in risky_terms)


def _tasks_from_understanding(
    query: str,
    context: SessionContext,
    item_id: str | None,
    understanding: UnderstandingResult,
) -> list[Task]:
    has_item_context = bool(item_id or context.current_item_id)
    contracts = [
        (need, _task_contract_for_need(need, has_item_context))
        for need in understanding.needs
    ]
    executable = [
        (need, contract)
        for need, contract in contracts
        if contract is not None
    ]
    if not executable:
        return []

    state = _semantic_state(query, context, item_id, understanding, executable)
    metadata_base = {_PLAN_STATE_KEY: _state_metadata(state)}
    task_ids = {
        need.need_id: f"q{index}"
        for index, (need, _contract) in enumerate(executable, start=1)
    }

    tasks: list[Task] = []
    for need, contract in executable:
        task_type, query_target, knowledge_scope = contract
        intent_context = {
            **need.intent_context(),
            "original_question": need.original_question,
            "normalized_question": need.normalized_question,
        }
        tasks.append(
            Task(
                task_ids[need.need_id],
                task_type,  # type: ignore[arg-type]
                need.original_question,
                {
                    **metadata_base,
                    "normalized_question": need.normalized_question,
                    "knowledge_scope": knowledge_scope,
                    "transaction_conditions": _transaction_conditions(need),
                    "intent_context": intent_context,
                    "reply_required": need.reply_required,
                },
                query_target=query_target,
                depends_on_task_ids=tuple(
                    task_ids[dependency]
                    for dependency in need.depends_on_need_ids
                    if dependency in task_ids
                ),
                execution_mode="xianyu_expert",
            )
        )
    return tasks


def _semantic_state(
    query: str,
    context: SessionContext,
    item_id: str | None,
    understanding: UnderstandingResult,
    executable: Sequence[tuple[UserNeed, tuple[str, str, str]]],
) -> PlannerState:
    remembered_order_id = session_order_id(
        {"order_id": context.current_order_id}
    ) or history_order_id(context.history)
    route, order_id = route_query(
        query,
        context.history,
        {"order_id": context.current_order_id},
    )
    has_item_context = bool(item_id or context.current_item_id)
    needs_item = any(
        _semantic_task_needs_item(need, contract, has_item_context)
        for need, contract in executable
    )
    question_plan = build_question_plan(query)
    return PlannerState(
        intent_match=IntentMatch(
            "SEMANTIC",
            (),
            "ai" if understanding.model_called else "rule",
        ),
        question_plan=question_plan,
        route=route,
        order_id=order_id,
        needs_item=needs_item,
        requires_item_resolution=needs_item and route == "rag",
        defer_to_order_context=remembered_order_id is not None,
        is_rule_followup=is_rule_followup(query, remembered_order_id),
        use_xianyu_without_item=bool(
            executable
            or context.current_item_id
            or item_id
            or is_seller_scoped_query(query)
            or _GREETING.fullmatch(query)
        ),
    )


def _task_contract_for_need(
    need: UserNeed,
    has_item_context: bool,
) -> tuple[str, str, str] | None:
    intent = need.intent
    if intent == "product.availability":
        return ("product", "availability.sale_status", "item_fact")
    if intent == "product.identity_model":
        return ("product", "identity.model", "item_fact")
    if intent == "product.sale_reason":
        return ("product", "product_info.sale_reason", "item_fact")
    if intent == "product.repair_history":
        return ("product", "history.repair_history", "item_fact")
    if intent == "product.condition_summary":
        return ("product", "condition.summary", "item_fact")
    if intent == "product.condition_issue":
        target = "function.shutter" if need.subject == "快门" else "condition.known_issues"
        return ("product", target, "item_fact")
    if intent == "product.lens_details":
        return ("product", "lens.details", "item_fact")
    if intent == "product.accessories":
        return ("product", "accessories.items", "item_fact")
    if intent == "product.function":
        target = "function.shutter" if "快门" in need.original_question else "function.overall"
        return ("product", target, "item_fact")
    if intent == "product.inspection_record":
        return ("product", "function.inspection_record", "item_fact")
    if intent == "product.model_knowledge":
        return ("product", "product.model_knowledge", "model_knowledge")

    if intent == "price.listed_price":
        return ("price", "price.listed_price", "item_fact")
    if intent == "price.minimum":
        return ("price", "price.minimum", "item_fact")
    if intent == "price.offer":
        return ("price", "price.offer", "item_fact")
    if intent == "price.additional_discount":
        return ("price", "price.additional_discount", "item_fact")
    if intent == "price.confirm":
        return ("price", "price.confirm", "item_fact")

    if intent == "shipping.dispatch_time":
        return (
            "service",
            "shipping.dispatch_time" if has_item_context else "seller_rule.general",
            "item_fact" if has_item_context else "seller_rule",
        )
    if intent == "shipping.carrier":
        return (
            "service",
            "shipping.carrier" if has_item_context else "seller_rule.general",
            "item_fact" if has_item_context else "seller_rule",
        )
    if intent == "shipping.fee":
        return (
            "service",
            "shipping.fee" if has_item_context else "seller_rule.general",
            "item_fact" if has_item_context else "seller_rule",
        )

    if intent in {
        "after_sale.consult",
        "after_sale.return_shipping_fee",
        "after_sale.refund_timing",
    }:
        return ("service", "seller_rule.general", "seller_rule")
    if intent == "service.greeting":
        return ("service", "greeting", "greeting")
    if intent == "service.thanks":
        return ("service", "thanks", "greeting")
    if intent == "service.no_reply":
        return ("service", "no_reply", "no_reply")
    return None


def _semantic_task_needs_item(
    need: UserNeed,
    contract: tuple[str, str, str],
    has_item_context: bool,
) -> bool:
    task_type, _query_target, knowledge_scope = contract
    return (
        task_type in {"product", "price"}
        or knowledge_scope == "item_fact"
        or (has_item_context and need.intent.startswith("after_sale."))
    )


def _transaction_conditions(need: UserNeed) -> dict[str, object]:
    conditions: dict[str, object] = {}
    request_kind = {
        "price.listed_price": "listed_price",
        "price.minimum": "minimum",
        "price.offer": "offer",
        "price.additional_discount": "additional_discount",
        "price.confirm": "confirm",
    }.get(need.intent)
    if request_kind is not None:
        conditions["request_kind"] = request_kind
    preconditions: list[dict[str, object]] = []
    for condition in need.conditions:
        condition_type = condition.get("type")
        if condition_type == "shipping":
            shipping = condition.get("value")
            if shipping in {"seller_pays", "buyer_pays"}:
                conditions["shipping"] = shipping
        elif condition_type == "amount":
            offer_cents = condition.get("offer_cents")
            if isinstance(offer_cents, int) and not isinstance(offer_cents, bool):
                conditions["offer_cents"] = offer_cents
        elif condition_type == "transaction_precondition":
            preconditions.append(dict(condition))
    if preconditions:
        conditions["preconditions"] = preconditions
    return conditions


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


def _looks_like_model_knowledge(query: str) -> bool:
    """Identify unplanned public model/technical questions for product fallback."""

    lowered = str(query or "").casefold()
    if re.search(r"[A-Za-z][A-Za-z0-9-]*(?:\s+[A-Za-z0-9][A-Za-z0-9-]*)+", query):
        return True
    return any(
        term in lowered
        for term in (
            "测光",
            "电池",
            "兼容",
            "卡口",
            "说明书",
            "操作",
            "使用方法",
            "参数",
            "规格",
            "系统",
        )
    )
