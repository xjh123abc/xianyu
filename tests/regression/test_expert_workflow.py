"""LangGraph expert workflow coverage for S2.5B."""

from __future__ import annotations

import asyncio

from app.services.chat_contracts import ChatMessage, SessionContext, Task, TaskResult
from app.services.workflows import ExpertTaskWorkflow


def _message() -> ChatMessage:
    return ChatMessage("xianyu", "seller", "chat", "buyer", "ITEM-001", "测试")


class _RecordingHandler:
    def __init__(self, prefix: str, *, status: str = "answered") -> None:
        self.prefix = prefix
        self.status = status
        self.calls: list[str] = []

    async def handle(
        self,
        task: Task,
        message: ChatMessage,
        context: SessionContext,
    ) -> TaskResult:
        del message, context
        self.calls.append(task.task_id)
        if self.status != "answered":
            return TaskResult(task.task_id, self.status, "", reason="forced_failure")
        return TaskResult(task.task_id, "answered", f"{self.prefix}:{task.query}")


class _SlowHandler:
    async def handle(
        self,
        task: Task,
        message: ChatMessage,
        context: SessionContext,
    ) -> TaskResult:
        del message, context
        await asyncio.sleep(0.05)
        return TaskResult(task.task_id, "answered", "too late")


def test_workflow_routes_only_the_existing_price_task() -> None:
    product = _RecordingHandler("product")
    price = _RecordingHandler("price")
    service = _RecordingHandler("service")
    workflow = ExpertTaskWorkflow(
        {"product": product, "price": price, "service": service}
    )

    results = asyncio.run(
        workflow.run(
            [Task("q1", "price", "最低多少？")],
            _message(),
            SessionContext(),
        )
    )

    assert results == [TaskResult("q1", "answered", "price:最低多少？")]
    assert product.calls == []
    assert price.calls == ["q1"]
    assert service.calls == []


def test_workflow_keeps_multiple_tasks_for_the_same_expert() -> None:
    product = _RecordingHandler("product")
    workflow = ExpertTaskWorkflow({"product": product})
    tasks = [
        Task("q1", "product", "修过吗？"),
        Task("q2", "product", "成色怎么样？"),
    ]

    results = asyncio.run(workflow.run(tasks, _message(), SessionContext()))

    assert product.calls == ["q1", "q2"]
    assert [result.task_id for result in results] == ["q1", "q2"]
    assert [result.answer for result in results] == [
        "product:修过吗？",
        "product:成色怎么样？",
    ]


def test_workflow_routes_product_price_and_service_nodes_once_needed() -> None:
    product = _RecordingHandler("product")
    price = _RecordingHandler("price")
    service = _RecordingHandler("service")
    workflow = ExpertTaskWorkflow(
        {"product": product, "price": price, "service": service}
    )
    tasks = [
        Task("q1", "product", "修过吗？"),
        Task("q2", "price", "最低多少？"),
        Task("q3", "service", "今天能发吗？"),
    ]

    results = asyncio.run(workflow.run(tasks, _message(), SessionContext()))

    assert product.calls == ["q1"]
    assert price.calls == ["q2"]
    assert service.calls == ["q3"]
    assert [result.task_id for result in results] == ["q1", "q2", "q3"]
    trace_nodes = [entry["node"] for entry in results[0].metadata["workflow_trace"]]
    assert trace_nodes == ["product_node", "price_node", "service_node"]


def test_workflow_runs_dependency_after_upstream_success() -> None:
    product = _RecordingHandler("product")
    price = _RecordingHandler("price")
    workflow = ExpertTaskWorkflow({"product": product, "price": price})
    tasks = [
        Task("q1", "product", "核验维修记录"),
        Task("q2", "price", "1900可以吗？", depends_on_task_ids=("q1",)),
    ]

    results = asyncio.run(workflow.run(tasks, _message(), SessionContext()))

    assert product.calls == ["q1"]
    assert price.calls == ["q2"]
    assert [result.status for result in results] == ["answered", "answered"]


def test_workflow_does_not_run_downstream_when_dependency_failed() -> None:
    product = _RecordingHandler("product", status="unavailable")
    price = _RecordingHandler("price")
    workflow = ExpertTaskWorkflow({"product": product, "price": price})
    tasks = [
        Task("q1", "product", "核验维修记录"),
        Task("q2", "price", "1900可以吗？", depends_on_task_ids=("q1",)),
    ]

    results = asyncio.run(workflow.run(tasks, _message(), SessionContext()))

    assert product.calls == ["q1"]
    assert price.calls == []
    assert results[1] == TaskResult(
        "q2",
        "unavailable",
        "",
        reason="dependency_unresolved:q1",
    )


def test_workflow_marks_dependency_cycle_invalid_without_calling_handlers() -> None:
    product = _RecordingHandler("product")
    price = _RecordingHandler("price")
    workflow = ExpertTaskWorkflow({"product": product, "price": price})
    tasks = [
        Task("q1", "product", "核验维修记录", depends_on_task_ids=("q2",)),
        Task("q2", "price", "1900可以吗？", depends_on_task_ids=("q1",)),
    ]

    results = asyncio.run(workflow.run(tasks, _message(), SessionContext()))

    assert product.calls == []
    assert price.calls == []
    assert [result.reason for result in results] == [
        "dependency_graph_invalid",
        "dependency_graph_invalid",
    ]


def test_workflow_routes_order_to_existing_capability_node() -> None:
    order = _RecordingHandler("order")
    workflow = ExpertTaskWorkflow({"order": order})

    results = asyncio.run(
        workflow.run(
            [Task("q1", "order", "查一下订单 TEST1001")],
            _message(),
            SessionContext(),
        )
    )

    assert order.calls == ["q1"]
    assert results[0].answer == "order:查一下订单 TEST1001"
    trace_nodes = [entry["node"] for entry in results[0].metadata["workflow_trace"]]
    assert trace_nodes == ["existing_capability_node"]


def test_workflow_uses_precomputed_response_without_calling_handler() -> None:
    product = _RecordingHandler("product")
    workflow = ExpertTaskWorkflow({"product": product})

    results = asyncio.run(
        workflow.run(
            [Task("q1", "product", "这台还在吗？")],
            _message(),
            SessionContext(),
            precomputed_responses={
                "q1": {
                    "action": "clarify",
                    "answer": "请先确认是哪一件商品。",
                    "can_answer": False,
                    "reason": "item_context_unavailable",
                }
            },
        )
    )

    assert product.calls == []
    assert results[0] == TaskResult(
        "q1",
        "clarify",
        "请先确认是哪一件商品。",
        reason="item_context_unavailable",
    )


def test_workflow_enforces_the_turn_deadline_around_handlers() -> None:
    workflow = ExpertTaskWorkflow({"service": _SlowHandler()}, budget_seconds=0.001)

    results = asyncio.run(
        workflow.run(
            [Task("q1", "service", "今天能发吗？")],
            _message(),
            SessionContext(),
        )
    )

    assert results[0] == TaskResult(
        "q1",
        "unavailable",
        "",
        reason="workflow_processing_timeout",
    )
