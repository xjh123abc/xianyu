"""Unified TaskResult adapters for the existing business capabilities.

S4 deliberately keeps the mature Xianyu expert orchestrator and order handler
intact.  This module only owns dispatch and the transport-free TaskResult
boundary that later ChatService stages can consume.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from typing import Protocol, cast

from app.services.chat_contracts import ChatMessage, SessionContext, Task, TaskResult
from app.services.chat_response import non_rag_response
from app.services.order_chat_handler import OrderChatHandler
from app.services.xianyu.expert_orchestrator import XianyuExpertOrchestrator
from app.services.xianyu.experts.contracts import (
    VALID_QUERY_TARGETS,
    ExpertContext,
    ExpertKnowledgeScope,
    ExpertName,
    ExpertResult,
    ExpertTask,
    QueryTarget,
)
from app.services.xianyu.knowledge_responder import XianyuKnowledgeResponder
from app.services.xianyu.responses import item_source


logger = logging.getLogger(__name__)

ItemLoader = Callable[[str], Awaitable[Mapping[str, object]]]
GeneralRagProvider = Callable[[], object]


class TaskHandler(Protocol):
    """One business capability behind the unified execution boundary."""

    async def handle(
        self,
        task: Task,
        message: ChatMessage,
        context: SessionContext,
    ) -> TaskResult: ...


class TaskExecutor:
    """Dispatch planner tasks without taking ownership of business rules."""

    def __init__(self, handlers: Mapping[str, TaskHandler]) -> None:
        self._handlers = dict(handlers)

    async def execute(
        self,
        tasks: Sequence[Task],
        message: ChatMessage,
        context: SessionContext,
        *,
        precomputed_responses: Mapping[str, Mapping[str, object]] | None = None,
    ) -> list[TaskResult]:
        """Execute every task and convert an isolated failure to TaskResult."""

        results: list[TaskResult] = []
        for task in tasks:
            completed = {result.task_id: result for result in results}
            try:
                dependencies = _task_dependencies(task)
            except ValueError:
                results.append(
                    TaskResult(
                        task.task_id,
                        "unavailable",
                        "",
                        reason="task_dependency_contract_invalid",
                    )
                )
                continue
            unresolved = [
                dependency
                for dependency in dependencies
                if dependency not in completed
                or completed[dependency].status != "answered"
            ]
            if unresolved:
                results.append(
                    TaskResult(
                        task.task_id,
                        "unavailable",
                        "",
                        reason="dependency_unresolved:" + ",".join(unresolved),
                    )
                )
                continue
            precomputed = (precomputed_responses or {}).get(task.task_id)
            if precomputed is not None:
                results.append(
                    _response_result(
                        task,
                        precomputed,
                        unavailable_reason="item_context_unavailable",
                    )
                )
                continue
            handler = self._handlers.get(task.task_type)
            if handler is None:
                results.append(
                    TaskResult(
                        task.task_id,
                        "unavailable",
                        "",
                        reason="task_handler_unavailable",
                    )
                )
                continue
            try:
                result = await handler.handle(task, message, context)
            except Exception:
                logger.exception(
                    "Task execution failed for task_id=%s type=%s",
                    task.task_id,
                    task.task_type,
                )
                results.append(
                    TaskResult(
                        task.task_id,
                        "unavailable",
                        "",
                        reason="task_execution_failed",
                    )
                )
                continue
            if result.task_id != task.task_id:
                results.append(
                    TaskResult(
                        task.task_id,
                        "unavailable",
                        "",
                        reason="task_result_contract_invalid",
                    )
                )
            else:
                results.append(result)
        return results


class XianyuExpertTaskHandler:
    """Execute Planner-owned expert tasks without entering legacy planning."""

    _ITEM_TASK_TYPES = frozenset({"product", "price"})
    _EXPERT_TASK_TYPES = frozenset({"product", "price", "service"})

    def __init__(
        self,
        *,
        expert_orchestrator: XianyuExpertOrchestrator,
        item_loader: ItemLoader,
    ) -> None:
        self._expert_orchestrator = expert_orchestrator
        self._item_loader = item_loader

    async def handle(
        self,
        task: Task,
        message: ChatMessage,
        context: SessionContext,
    ) -> TaskResult:
        if task.task_type not in self._EXPERT_TASK_TYPES:
            return TaskResult(task.task_id, "unavailable", "", reason="task_type_mismatch")

        item_id = message.item_id or context.current_item_id
        if task.task_type in self._ITEM_TASK_TYPES and item_id is None:
            return TaskResult(
                task.task_id,
                "unavailable",
                "",
                reason="item_context_unavailable",
            )
        item = (
            await self._load_item(task, item_id, context)
            if task.task_type in self._ITEM_TASK_TYPES
            else self._resolved_context_item(item_id, context)
        )
        if isinstance(item, TaskResult):
            return item

        try:
            expert_task = to_expert_task(task)
        except ValueError:
            logger.exception("Invalid expert task contract for task_id=%s", task.task_id)
            return TaskResult(
                task.task_id,
                "unavailable",
                "",
                reason="expert_task_contract_invalid",
            )
        # The unified TaskExecutor has already resolved outer dependencies.
        expert_task = replace(expert_task, depends_on_task_ids=())
        expert_context = ExpertContext(
            query=task.query,
            item=item,
            history=context.history,
            xianyu_context=context.platform_context.get("xianyu", {}),
        )
        results = await self._expert_orchestrator.execute_tasks(
            [expert_task],
            expert_context,
        )
        if len(results) != 1 or results[0].task_id != task.task_id:
            return TaskResult(
                task.task_id,
                "unavailable",
                "",
                reason="expert_result_contract_invalid",
            )
        return _expert_result(task, results[0], item)

    async def _load_item(
        self,
        task: Task,
        item_id: str | None,
        context: SessionContext,
    ) -> Mapping[str, object] | TaskResult | None:
        if item_id is None:
            return None
        xianyu_context = context.platform_context.get("xianyu", {})
        resolved_item = xianyu_context.get("resolved_item")
        if (
            isinstance(resolved_item, Mapping)
            and resolved_item.get("found") is True
            and str(resolved_item.get("item_id")) == item_id
        ):
            return resolved_item
        try:
            item = await self._item_loader(item_id)
        except Exception:
            logger.exception("Item lookup failed for task_id=%s", task.task_id)
            return TaskResult(task.task_id, "unavailable", "", reason="item_lookup_failed")
        if not isinstance(item, Mapping) or item.get("found") is not True:
            return TaskResult(
                task.task_id,
                "unavailable",
                "",
                reason="item_context_unavailable",
            )
        return item

    @staticmethod
    def _resolved_context_item(
        item_id: str | None,
        context: SessionContext,
    ) -> Mapping[str, object] | None:
        """Service tasks reuse an already validated item but never retry MCP."""

        resolved_item = context.platform_context.get("xianyu", {}).get("resolved_item")
        if (
            isinstance(resolved_item, Mapping)
            and resolved_item.get("found") is True
            and item_id is not None
            and str(resolved_item.get("item_id")) == item_id
        ):
            return resolved_item
        return None


def to_expert_task(task: Task) -> ExpertTask:
    """Convert the unified Planner contract to one executable expert task."""

    if task.task_type not in XianyuExpertTaskHandler._EXPERT_TASK_TYPES:
        raise ValueError("only product, price, and service tasks have expert contracts")
    normalized_question = _optional_text(task.metadata.get("normalized_question"))
    knowledge_scope = _optional_text(task.metadata.get("knowledge_scope"))
    query_target = task.query_target
    transaction_conditions = task.metadata.get("transaction_conditions", {})
    dependencies = _task_dependencies(task)
    if normalized_question is None:
        raise ValueError("expert task is missing normalized_question")
    if knowledge_scope not in {"item_fact", "model_knowledge", "seller_rule", "greeting"}:
        raise ValueError("expert task contains an invalid knowledge_scope")
    if query_target not in VALID_QUERY_TARGETS:
        raise ValueError("expert task contains an invalid query_target")
    if not isinstance(transaction_conditions, Mapping):
        raise ValueError("expert task transaction_conditions must be a mapping")
    return ExpertTask(
        task_id=task.task_id,
        expert=cast(ExpertName, task.task_type),
        question_fragment=task.query,
        normalized_question=normalized_question,
        knowledge_scope=cast(ExpertKnowledgeScope, knowledge_scope),
        query_target=cast(QueryTarget, query_target),
        transaction_conditions=dict(transaction_conditions),
        depends_on_task_ids=dependencies,
        original_question=task.query,
    )


def _task_dependencies(task: Task) -> tuple[str, ...]:
    """Read the unified scheduler contract without inspecting metadata."""

    return task.depends_on_task_ids


def _expert_result(
    task: Task,
    result: ExpertResult,
    item: Mapping[str, object] | None,
) -> TaskResult:
    sources = [dict(source) for source in result.sources if isinstance(source, Mapping)]
    if item is not None:
        item_evidence = item_source(item)
        if item_evidence not in sources:
            sources.insert(0, item_evidence)
    if result.status == "answered" and isinstance(result.answer, str) and result.answer.strip():
        # Do not normalize or replace model text after a successful response.
        answer = result.answer
        response: dict[str, object] = {
            "query": task.query,
            "route": "xianyu",
            "action": "reply",
            "answer": answer,
            "sources": sources,
            "results": [],
            "reliability": None,
            "next_step": None,
            "can_answer": True,
            "evidence": result.evidence,
            "raw_answer": result.raw_answer or answer,
        }
        if item is not None:
            response.update({"item_id": item["item_id"], "item_info": dict(item)})
        return TaskResult(
            task.task_id,
            "answered",
            answer,
            sources,
            metadata={"response": response},
        )
    reason = result.reason or "expert_answer_unavailable"
    answer = result.answer if isinstance(result.answer, str) else ""
    response = {
        "query": task.query,
        "route": "xianyu",
        "action": "reply",
        "answer": answer,
        "sources": sources,
        "results": [],
        "reliability": None,
        "next_step": None,
        "can_answer": False,
        "reason": reason,
    }
    if item is not None:
        response.update({"item_id": item["item_id"], "item_info": dict(item)})
    return TaskResult(
        task.task_id,
        "unavailable",
        answer,
        sources,
        reason,
        metadata={"response": response},
    )


class OrderTaskHandler:
    """Adapt the existing order handler through its standalone MCP path."""

    def __init__(self, *, order_handler: OrderChatHandler) -> None:
        self._order_handler = order_handler

    async def handle(
        self,
        task: Task,
        message: ChatMessage,
        context: SessionContext,
    ) -> TaskResult:
        del message
        state = _planner_metadata(task)
        order_id = _optional_text(state.get("order_id")) or context.current_order_id
        if task.execution_mode == "missing_order_id" or order_id is None:
            response = non_rag_response(
                task.query,
                "请提供订单号，并重新发送完整问题，例如：帮我查订单 TEST1001。",
                can_answer=False,
                route="unified",
                action="reply",
            )
            response["reason"] = "order_id_missing"
            return _response_result(task, response, unavailable_reason="order_answer_unavailable")
        response = await self._order_handler.order(task.query, order_id)
        return _response_result(task, response, unavailable_reason="order_answer_unavailable")


def _planner_metadata(task: Task) -> Mapping[str, object]:
    value = task.metadata.get("_planner_state")
    return value if isinstance(value, Mapping) else {}


def _optional_text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _response_result(
    task: Task,
    response: object,
    *,
    unavailable_reason: str,
) -> TaskResult:
    if not isinstance(response, Mapping):
        return TaskResult(task.task_id, "unavailable", "", reason=unavailable_reason)
    sources = response.get("sources")
    safe_sources = [dict(source) for source in sources if isinstance(source, Mapping)] if isinstance(sources, list) else []
    answer = response.get("answer")
    response_reason = _optional_text(response.get("reason"))
    if (
        response.get("action") == "clarify"
        and response_reason != "knowledge_evidence_unavailable"
        and isinstance(answer, str)
        and answer.strip()
    ):
        return TaskResult(
            task.task_id,
            "clarify",
            answer.strip(),
            safe_sources,
            response_reason or unavailable_reason,
            metadata={"response": dict(response)},
        )
    if (
        response.get("can_answer") is True
        or (
            response.get("action") == "reply"
            and response.get("can_answer") is not False
        )
    ) and isinstance(answer, str) and answer.strip():
        return TaskResult(
            task.task_id,
            "answered",
            answer.strip(),
            safe_sources,
            metadata={"response": dict(response)},
        )
    reason = response_reason or unavailable_reason
    return TaskResult(
        task.task_id,
        "unavailable",
        answer if isinstance(answer, str) else "",
        safe_sources,
        reason,
        metadata={"response": dict(response)},
    )


class ServiceTaskHandler:
    """Use the knowledge boundary for the service half of a combined turn."""

    def __init__(
        self,
        *,
        expert_handler: XianyuExpertTaskHandler,
        knowledge_responder: XianyuKnowledgeResponder,
        general_rag: GeneralRagProvider,
    ) -> None:
        self._expert_handler = expert_handler
        self._knowledge_responder = knowledge_responder
        self._general_rag = general_rag

    async def handle(
        self,
        task: Task,
        message: ChatMessage,
        context: SessionContext,
    ) -> TaskResult:
        execution_mode = task.execution_mode
        if execution_mode == "common_knowledge":
            response = await self._knowledge_responder.handle_common(
                task.query,
            )
            return _response_result(task, response, unavailable_reason="service_answer_unavailable")
        if execution_mode == "general_rag":
            chat = getattr(self._general_rag(), "chat", None)
            if not callable(chat):
                return TaskResult(
                    task.task_id,
                    "unavailable",
                    "",
                    reason="general_rag_handler_unavailable",
                )
            response = await asyncio.to_thread(chat, task.query)
            return _response_result(task, response, unavailable_reason="general_rag_answer_unavailable")
        if execution_mode == "unsupported_action":
            response = non_rag_response(
                task.query,
                "本版本暂不支持取消订单、退款或修改地址等操作，仅支持订单状态查询。",
                can_answer=False,
                route="unified",
                action="reply",
            )
            response["reason"] = "unsupported_action"
            return _response_result(task, response, unavailable_reason="unsupported_action")
        return await self._expert_handler.handle(task, message, context)
