"""LangGraph workflow for executing planner-owned expert tasks.

The graph owns one buyer turn only: task readiness, node routing, dependency
handling, result collection, and trace metadata. Domain answers still come from
the existing task handlers and expert agents.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from app.services.chat_contracts import ChatMessage, SessionContext, Task, TaskResult
from config.settings import settings


logger = logging.getLogger(__name__)


class WorkflowTaskHandler(Protocol):
    """A single business capability node behind the graph boundary."""

    async def handle(
        self,
        task: Task,
        message: ChatMessage,
        context: SessionContext,
    ) -> TaskResult: ...


class ExpertWorkflowState(TypedDict):
    """Per-turn graph state; it does not replace session memory."""

    run_id: str
    message: ChatMessage
    context: SessionContext
    tasks: tuple[Task, ...]
    pending_task_ids: tuple[str, ...]
    ready_task_ids: tuple[str, ...]
    active_task_ids: tuple[str, ...]
    results_by_id: dict[str, TaskResult]
    precomputed_responses: dict[str, Mapping[str, object]]
    deadline: float | None
    trace: list[dict[str, object]]
    next_node: str


class ExpertTaskWorkflow:
    """Execute a Task DAG through LangGraph nodes and return TaskResult values."""

    def __init__(
        self,
        handlers: Mapping[str, WorkflowTaskHandler],
        *,
        budget_seconds: float | None = None,
    ) -> None:
        self._handlers = dict(handlers)
        self._budget_seconds = (
            _default_budget_seconds() if budget_seconds is None else budget_seconds
        )
        if self._budget_seconds is not None and self._budget_seconds <= 0:
            raise ValueError("budget_seconds must be greater than zero")
        self._graph = self._build_graph().compile()
        self.last_trace: list[dict[str, object]] = []

    async def run(
        self,
        tasks: Sequence[Task],
        message: ChatMessage,
        context: SessionContext,
        *,
        precomputed_responses: Mapping[str, Mapping[str, object]] | None = None,
    ) -> list[TaskResult]:
        """Run the graph once for the already-planned tasks."""

        ordered_tasks = tuple(tasks)
        if not ordered_tasks:
            self.last_trace = []
            return []
        deadline = (
            None
            if self._budget_seconds is None
            else time.monotonic() + self._budget_seconds
        )
        initial_state: ExpertWorkflowState = {
            "run_id": uuid.uuid4().hex,
            "message": message,
            "context": context,
            "tasks": ordered_tasks,
            "pending_task_ids": tuple(task.task_id for task in ordered_tasks),
            "ready_task_ids": (),
            "active_task_ids": (),
            "results_by_id": {},
            "precomputed_responses": {
                task_id: dict(response)
                for task_id, response in (precomputed_responses or {}).items()
            },
            "deadline": deadline,
            "trace": [],
            "next_node": "prepare_ready_tasks",
        }
        final_state = await self._graph.ainvoke(initial_state)
        trace = list(final_state.get("trace", []))
        self.last_trace = trace
        results_by_id = dict(final_state.get("results_by_id", {}))
        return [
            _with_workflow_trace(
                results_by_id.get(task.task_id)
                or TaskResult(
                    task.task_id,
                    "unavailable",
                    "",
                    reason="task_missing_result",
                ),
                trace,
            )
            for task in ordered_tasks
        ]

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(ExpertWorkflowState)
        graph.add_node("prepare_ready_tasks", self._prepare_ready_tasks)
        graph.add_node("product_node", self._product_node)
        graph.add_node("price_node", self._price_node)
        graph.add_node("service_node", self._service_node)
        graph.add_node("existing_capability_node", self._existing_capability_node)
        graph.add_node("record_results", self._record_results)
        graph.add_node("finish_or_continue", self._finish_or_continue)

        graph.add_edge(START, "prepare_ready_tasks")
        graph.add_conditional_edges(
            "prepare_ready_tasks",
            _next_node,
            {
                "product_node": "product_node",
                "price_node": "price_node",
                "service_node": "service_node",
                "existing_capability_node": "existing_capability_node",
                "finish_or_continue": "finish_or_continue",
            },
        )
        graph.add_edge("product_node", "record_results")
        graph.add_edge("price_node", "record_results")
        graph.add_edge("service_node", "record_results")
        graph.add_edge("existing_capability_node", "record_results")
        graph.add_edge("record_results", "finish_or_continue")
        graph.add_conditional_edges(
            "finish_or_continue",
            _continue_or_end,
            {
                "prepare_ready_tasks": "prepare_ready_tasks",
                "end": END,
            },
        )
        return graph

    async def _prepare_ready_tasks(
        self,
        state: ExpertWorkflowState,
    ) -> ExpertWorkflowState:
        updated = _copy_state(state)
        updated["ready_task_ids"] = ()
        updated["active_task_ids"] = ()

        while updated["pending_task_ids"]:
            if _deadline_expired(updated["deadline"]):
                for task in _pending_tasks(updated):
                    _record_result(
                        updated,
                        task,
                        TaskResult(
                            task.task_id,
                            "unavailable",
                            "",
                            reason="workflow_processing_timeout",
                        ),
                        "prepare_ready_tasks",
                        0.0,
                    )
                updated["pending_task_ids"] = ()
                updated["next_node"] = "finish_or_continue"
                return updated

            changed = _mark_failed_dependencies(updated)
            if changed:
                continue

            precomputed = _consume_precomputed_ready_tasks(updated)
            if precomputed:
                continue

            ready = _ready_tasks(updated)
            if ready:
                route = _route_node(ready[0])
                active = tuple(task.task_id for task in ready if _route_node(task) == route)
                updated["ready_task_ids"] = tuple(task.task_id for task in ready)
                updated["active_task_ids"] = active
                updated["next_node"] = route
                return updated

            _close_invalid_or_unresolved_graph(updated)
            updated["next_node"] = "finish_or_continue"
            return updated

        updated["next_node"] = "finish_or_continue"
        return updated

    async def _product_node(self, state: ExpertWorkflowState) -> ExpertWorkflowState:
        return await self._execute_active(state, "product_node", {"product"})

    async def _price_node(self, state: ExpertWorkflowState) -> ExpertWorkflowState:
        return await self._execute_active(state, "price_node", {"price"})

    async def _service_node(self, state: ExpertWorkflowState) -> ExpertWorkflowState:
        return await self._execute_active(state, "service_node", {"service"})

    async def _existing_capability_node(
        self,
        state: ExpertWorkflowState,
    ) -> ExpertWorkflowState:
        return await self._execute_active(
            state,
            "existing_capability_node",
            set(),
            include_other_types=True,
        )

    async def _execute_active(
        self,
        state: ExpertWorkflowState,
        node_name: str,
        allowed_task_types: set[str],
        *,
        include_other_types: bool = False,
    ) -> ExpertWorkflowState:
        updated = _copy_state(state)
        task_by_id = _task_by_id(updated)
        for task_id in updated["active_task_ids"]:
            task = task_by_id.get(task_id)
            if task is None:
                continue
            if not include_other_types and task.task_type not in allowed_task_types:
                _record_result(
                    updated,
                    task,
                    TaskResult(
                        task.task_id,
                        "unavailable",
                        "",
                        reason="workflow_route_mismatch",
                    ),
                    node_name,
                    0.0,
                )
                continue
            result = await self._run_handler(task, updated, node_name)
            _record_result(updated, task, result, node_name, result.metadata.get("_duration_ms"))
        return updated

    async def _run_handler(
        self,
        task: Task,
        state: ExpertWorkflowState,
        node_name: str,
    ) -> TaskResult:
        timeout_seconds = _remaining_seconds(state["deadline"])
        if timeout_seconds is not None and timeout_seconds <= 0:
            return TaskResult(
                task.task_id,
                "unavailable",
                "",
                reason="workflow_processing_timeout",
            )
        handler = self._handlers.get(task.task_type)
        if handler is None:
            return TaskResult(
                task.task_id,
                "unavailable",
                "",
                reason="task_handler_unavailable",
            )
        started = time.perf_counter()
        try:
            coroutine = handler.handle(task, state["message"], state["context"])
            result = (
                await coroutine
                if timeout_seconds is None
                else await asyncio.wait_for(coroutine, timeout=timeout_seconds)
            )
        except asyncio.TimeoutError:
            return TaskResult(
                task.task_id,
                "unavailable",
                "",
                reason="workflow_processing_timeout",
                metadata={"_duration_ms": _elapsed_ms(started)},
            )
        except Exception:
            logger.exception(
                "Workflow node %s failed for task_id=%s type=%s",
                node_name,
                task.task_id,
                task.task_type,
            )
            return TaskResult(
                task.task_id,
                "unavailable",
                "",
                reason="task_execution_failed",
                metadata={"_duration_ms": _elapsed_ms(started)},
            )
        if result.task_id != task.task_id:
            return TaskResult(
                task.task_id,
                "unavailable",
                "",
                reason="task_result_contract_invalid",
                metadata={"_duration_ms": _elapsed_ms(started)},
            )
        return _with_duration(result, _elapsed_ms(started))

    async def _record_results(
        self,
        state: ExpertWorkflowState,
    ) -> ExpertWorkflowState:
        updated = _copy_state(state)
        active = set(updated["active_task_ids"])
        updated["pending_task_ids"] = tuple(
            task_id for task_id in updated["pending_task_ids"] if task_id not in active
        )
        updated["active_task_ids"] = ()
        return updated

    async def _finish_or_continue(
        self,
        state: ExpertWorkflowState,
    ) -> ExpertWorkflowState:
        updated = _copy_state(state)
        updated["next_node"] = (
            "prepare_ready_tasks" if updated["pending_task_ids"] else "end"
        )
        return updated


def _next_node(state: ExpertWorkflowState) -> str:
    return state["next_node"]


def _continue_or_end(state: ExpertWorkflowState) -> str:
    return state["next_node"]


def _copy_state(state: ExpertWorkflowState) -> ExpertWorkflowState:
    return {
        **state,
        "pending_task_ids": tuple(state["pending_task_ids"]),
        "ready_task_ids": tuple(state["ready_task_ids"]),
        "active_task_ids": tuple(state["active_task_ids"]),
        "results_by_id": dict(state["results_by_id"]),
        "precomputed_responses": dict(state["precomputed_responses"]),
        "trace": [dict(entry) for entry in state["trace"]],
    }


def _task_by_id(state: ExpertWorkflowState) -> dict[str, Task]:
    return {task.task_id: task for task in state["tasks"]}


def _pending_tasks(state: ExpertWorkflowState) -> list[Task]:
    by_id = _task_by_id(state)
    return [by_id[task_id] for task_id in state["pending_task_ids"] if task_id in by_id]


def _ready_tasks(state: ExpertWorkflowState) -> list[Task]:
    return [
        task
        for task in _pending_tasks(state)
        if all(
            dependency in state["results_by_id"]
            and state["results_by_id"][dependency].status == "answered"
            for dependency in task.depends_on_task_ids
        )
    ]


def _mark_failed_dependencies(state: ExpertWorkflowState) -> bool:
    changed = False
    for task in list(_pending_tasks(state)):
        failed_dependencies = [
            dependency
            for dependency in task.depends_on_task_ids
            if dependency in state["results_by_id"]
            and state["results_by_id"][dependency].status != "answered"
        ]
        if not failed_dependencies:
            continue
        _record_result(
            state,
            task,
            TaskResult(
                task.task_id,
                "unavailable",
                "",
                reason="dependency_unresolved:" + ",".join(failed_dependencies),
            ),
            "prepare_ready_tasks",
            0.0,
        )
        state["pending_task_ids"] = tuple(
            task_id
            for task_id in state["pending_task_ids"]
            if task_id != task.task_id
        )
        changed = True
    return changed


def _consume_precomputed_ready_tasks(state: ExpertWorkflowState) -> bool:
    changed = False
    for task in list(_ready_tasks(state)):
        response = state["precomputed_responses"].get(task.task_id)
        if response is None:
            continue
        _record_result(
            state,
            task,
            _response_result(
                task,
                response,
                unavailable_reason="item_context_unavailable",
            ),
            "prepare_ready_tasks",
            0.0,
        )
        state["pending_task_ids"] = tuple(
            task_id
            for task_id in state["pending_task_ids"]
            if task_id != task.task_id
        )
        changed = True
    return changed


def _close_invalid_or_unresolved_graph(state: ExpertWorkflowState) -> None:
    task_ids = set(_task_by_id(state))
    completed = set(state["results_by_id"])
    for task in _pending_tasks(state):
        missing = [
            dependency
            for dependency in task.depends_on_task_ids
            if dependency not in task_ids and dependency not in completed
        ]
        reason = (
            "dependency_unresolved:" + ",".join(missing)
            if missing
            else "dependency_graph_invalid"
        )
        _record_result(
            state,
            task,
            TaskResult(task.task_id, "unavailable", "", reason=reason),
            "prepare_ready_tasks",
            0.0,
        )
    state["pending_task_ids"] = ()


def _route_node(task: Task) -> str:
    if task.task_type == "product":
        return "product_node"
    if task.task_type == "price":
        return "price_node"
    if task.task_type == "service":
        return "service_node"
    return "existing_capability_node"


def _record_result(
    state: ExpertWorkflowState,
    task: Task,
    result: TaskResult,
    node_name: str,
    duration_ms: object,
) -> None:
    safe_duration = (
        float(duration_ms)
        if isinstance(duration_ms, (int, float)) and not isinstance(duration_ms, bool)
        else 0.0
    )
    state["results_by_id"][task.task_id] = result
    state["trace"].append(
        {
            "run_id": state["run_id"],
            "node": node_name,
            "task_id": task.task_id,
            "task_type": task.task_type,
            "execution_mode": task.execution_mode,
            "query_target": task.query_target,
            "status": result.status,
            "reason": result.reason,
            "duration_ms": round(safe_duration, 3),
            "sources": [dict(source) for source in result.sources],
            "raw_output_recorded": _has_raw_output(result),
        }
    )


def _with_duration(result: TaskResult, duration_ms: float) -> TaskResult:
    return TaskResult(
        result.task_id,
        result.status,
        result.answer,
        result.sources,
        result.reason,
        metadata={**result.metadata, "_duration_ms": duration_ms},
    )


def _with_workflow_trace(
    result: TaskResult,
    trace: Sequence[Mapping[str, object]],
) -> TaskResult:
    return TaskResult(
        result.task_id,
        result.status,
        result.answer,
        result.sources,
        result.reason,
        metadata={
            **result.metadata,
            "workflow_trace": [dict(entry) for entry in trace],
        },
    )


def _response_result(
    task: Task,
    response: object,
    *,
    unavailable_reason: str,
) -> TaskResult:
    if not isinstance(response, Mapping):
        return TaskResult(task.task_id, "unavailable", "", reason=unavailable_reason)
    sources = response.get("sources")
    safe_sources = (
        [dict(source) for source in sources if isinstance(source, Mapping)]
        if isinstance(sources, list)
        else []
    )
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


def _optional_text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _has_raw_output(result: TaskResult) -> bool:
    response = result.metadata.get("response")
    return isinstance(response, Mapping) and any(
        isinstance(response.get(key), str) and response.get(key)
        for key in ("raw_answer", "evidence")
    )


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _deadline_expired(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() >= deadline


def _remaining_seconds(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    return deadline - time.monotonic()


def _default_budget_seconds() -> float | None:
    if settings is None:
        return 25.0
    configured_budget = float(getattr(settings, "xianyu_expert_budget_seconds", 25.0))
    channel_timeout = float(getattr(settings, "xianyu_chat_api_timeout_seconds", 30.0))
    return min(configured_budget, channel_timeout * 0.9)
