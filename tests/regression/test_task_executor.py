"""S4 acceptance tests for unified TaskResult execution adapters."""

from __future__ import annotations

import asyncio

from app.services.chat_contracts import ChatMessage, SessionContext, Task, TaskResult
from app.services.task_executor import (
    OrderTaskHandler,
    TaskExecutor,
    XianyuExpertTaskHandler,
)


def _message(*, item_id: str | None = None) -> ChatMessage:
    return ChatMessage("xianyu", "seller", "chat", "buyer", item_id, "测试问题")


class _RecordingHandler:
    def __init__(self, result: TaskResult) -> None:
        self.result = result
        self.calls: list[str] = []

    async def handle(
        self,
        task: Task,
        message: ChatMessage,
        context: SessionContext,
    ) -> TaskResult:
        del message, context
        self.calls.append(task.task_id)
        return self.result


def test_task_executor_dispatches_each_task_type_to_its_registered_handler() -> None:
    product = _RecordingHandler(TaskResult("product-1", "answered", "商品事实"))
    price = _RecordingHandler(TaskResult("price-1", "answered", "价格事实"))
    executor = TaskExecutor({"product": product, "price": price})

    results = asyncio.run(
        executor.execute(
            [Task("product-1", "product", "修过吗"), Task("price-1", "price", "最低多少")],
            _message(),
            SessionContext(),
        )
    )

    assert [result.status for result in results] == ["answered", "answered"]
    assert product.calls == ["product-1"]
    assert price.calls == ["price-1"]


def test_task_executor_isolates_a_single_handler_failure() -> None:
    class _FailingHandler:
        async def handle(
            self,
            task: Task,
            message: ChatMessage,
            context: SessionContext,
        ) -> TaskResult:
            del task, message, context
            raise RuntimeError("expected test failure")

    result = asyncio.run(
        TaskExecutor({"service": _FailingHandler()}).execute(
            [Task("service-1", "service", "多久发货")],
            _message(),
            SessionContext(),
        )
    )

    assert result == [
        TaskResult("service-1", "unavailable", "", reason="task_execution_failed")
    ]


def test_xianyu_expert_handler_adapts_existing_expert_response_to_task_result() -> None:
    class _ExpertOrchestrator:
        async def handle(self, query: str, **kwargs: object) -> dict[str, object]:
            assert query == "这个修过吗？"
            assert kwargs["item"] == {"found": True, "item_id": "ITEM-001"}
            return {
                "action": "reply",
                "answer": "没有维修记录。",
                "sources": [{"source": "mcp:get_item_info", "index": "ITEM-001"}],
            }

    async def _load_item(item_id: str) -> dict[str, object]:
        assert item_id == "ITEM-001"
        return {"found": True, "item_id": item_id}

    result = asyncio.run(
        XianyuExpertTaskHandler(
            expert_orchestrator=_ExpertOrchestrator(),  # type: ignore[arg-type]
            item_loader=_load_item,
        ).handle(
            Task("product-1", "product", "这个修过吗？"),
            _message(item_id="ITEM-001"),
            SessionContext(),
        )
    )

    assert result == TaskResult(
        "product-1",
        "answered",
        "没有维修记录。",
        [{"source": "mcp:get_item_info", "index": "ITEM-001"}],
    )


def test_order_handler_keeps_the_existing_combined_order_route() -> None:
    class _OrderHandler:
        def __init__(self) -> None:
            self.combined_calls: list[tuple[str, str, list[dict[str, str]]]] = []

        async def combined(
            self,
            query: str,
            order_id: str,
            history: list[dict[str, str]],
        ) -> dict[str, object]:
            self.combined_calls.append((query, order_id, history))
            return {"action": "reply", "answer": "订单正在运输中。"}

        async def order(self, query: str, order_id: str) -> dict[str, object]:
            raise AssertionError(f"unexpected order call: {query} {order_id}")

    handler = _OrderHandler()
    task = Task(
        "order-1",
        "order",
        "TEST1001 怎么还没到？一般多久到？",
        {"_planner_state": {"route": "rag_mcp", "order_id": "TEST1001"}},
    )
    context = SessionContext(history=[{"role": "user", "content": "TEST1001"}])

    result = asyncio.run(OrderTaskHandler(order_handler=handler).handle(task, _message(), context))  # type: ignore[arg-type]

    assert result == TaskResult("order-1", "answered", "订单正在运输中。")
    assert handler.combined_calls == [(task.query, "TEST1001", context.history)]
