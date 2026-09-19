"""Unified TaskResult adapters for the existing business capabilities.

S4 deliberately keeps the mature Xianyu expert orchestrator and order handler
intact.  This module only owns dispatch and the transport-free TaskResult
boundary that later ChatService stages can consume.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Protocol

from app.services.chat_contracts import ChatMessage, SessionContext, Task, TaskResult
from app.services.order_chat_handler import OrderChatHandler
from app.services.xianyu.knowledge_responder import XianyuKnowledgeResponder
from app.services.xianyu.expert_orchestrator import XianyuExpertOrchestrator


logger = logging.getLogger(__name__)

ItemLoader = Callable[[str], Awaitable[Mapping[str, object]]]


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
    ) -> list[TaskResult]:
        """Execute every task and convert an isolated failure to TaskResult."""

        results: list[TaskResult] = []
        for task in tasks:
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
    """Adapt the existing Product/Price/Service expert path to TaskResult."""

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
        item = await self._load_item(task, item_id)
        if isinstance(item, TaskResult):
            return item

        response = await self._expert_orchestrator.handle(
            task.query,
            item=item,
            history=context.history,
            session_state={"xianyu_context": context.platform_context.get("xianyu", {})},
        )
        return _response_result(task, response, unavailable_reason="expert_answer_unavailable")

    async def _load_item(
        self,
        task: Task,
        item_id: str | None,
    ) -> Mapping[str, object] | TaskResult | None:
        if item_id is None:
            return None
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
        route = state.get("route")
        order_id = _optional_text(state.get("order_id")) or context.current_order_id
        if route == "missing_order_id" or order_id is None:
            return TaskResult(task.task_id, "unavailable", "", reason="order_id_missing")
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
    if (
        response.get("action") == "reply" or response.get("can_answer") is True
    ) and isinstance(answer, str) and answer.strip():
        return TaskResult(
            task.task_id,
            "answered",
            answer.strip(),
            safe_sources,
            metadata={"response": dict(response)},
        )
    reason = _optional_text(response.get("reason")) or unavailable_reason
    return TaskResult(
        task.task_id,
        "unavailable",
        "",
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
    ) -> None:
        self._expert_handler = expert_handler
        self._knowledge_responder = knowledge_responder

    async def handle(
        self,
        task: Task,
        message: ChatMessage,
        context: SessionContext,
    ) -> TaskResult:
        if _planner_metadata(task).get("route") != "rag_mcp":
            return await self._expert_handler.handle(task, message, context)
        response = await self._knowledge_responder.handle_common(task.query)
        return _response_result(task, response, unavailable_reason="service_answer_unavailable")
