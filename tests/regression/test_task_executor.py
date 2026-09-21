"""S4 acceptance tests for unified TaskResult execution adapters."""

from __future__ import annotations

import asyncio

from app.services.chat_contracts import ChatMessage, SessionContext, Task, TaskResult
from app.services.task_executor import (
    OrderTaskHandler,
    ServiceTaskHandler,
    TaskExecutor,
    XianyuExpertTaskHandler,
    to_expert_task,
)
from app.services.xianyu.experts.contracts import ExpertResult


def _message(*, item_id: str | None = None) -> ChatMessage:
    return ChatMessage("xianyu", "seller", "chat", "buyer", item_id, "测试问题")


def _expert_task(
    task_id: str,
    task_type: str,
    query: str,
    *,
    query_target: str,
    knowledge_scope: str = "item_fact",
) -> Task:
    return Task(
        task_id,
        task_type,  # type: ignore[arg-type]
        query,
        {
            "normalized_question": query,
            "knowledge_scope": knowledge_scope,
            "transaction_conditions": {},
        },
        query_target=query_target,
        execution_mode="xianyu_expert",
    )


def test_to_expert_task_preserves_explicit_identity_target_and_dependencies() -> None:
    task = Task(
        "price-2",
        "price",
        "如果没修过，最低多少？",
        {
            "normalized_question": "最低价",
            "knowledge_scope": "item_fact",
            "transaction_conditions": {"request_kind": "minimum"},
        },
        query_target="price.minimum",
        depends_on_task_ids=("product-1",),
        execution_mode="xianyu_expert",
    )

    expert_task = to_expert_task(task)

    assert expert_task.task_id == task.task_id
    assert expert_task.expert == task.task_type
    assert expert_task.original_question == task.query
    assert expert_task.query_target == task.query_target
    assert expert_task.depends_on_task_ids == task.depends_on_task_ids
    assert expert_task.transaction_conditions == {"request_kind": "minimum"}


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
        async def execute_tasks(self, tasks, context):
            assert context.query == "这个修过吗？"
            assert context.item == {"found": True, "item_id": "ITEM-001"}
            assert tasks[0].query_target == "history.repair_history"
            return [
                ExpertResult.answered(
                    tasks[0],
                    "没有维修记录。",
                    sources=[{"source": "mcp:get_item_info", "index": "ITEM-001"}],
                )
            ]

    async def _load_item(item_id: str) -> dict[str, object]:
        assert item_id == "ITEM-001"
        return {"found": True, "item_id": item_id}

    result = asyncio.run(
        XianyuExpertTaskHandler(
            expert_orchestrator=_ExpertOrchestrator(),  # type: ignore[arg-type]
            item_loader=_load_item,
        ).handle(
            _expert_task(
                "product-1",
                "product",
                "这个修过吗？",
                query_target="history.repair_history",
            ),
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


def test_xianyu_expert_handler_uses_each_task_query_without_reusing_combined_answer() -> None:
    class _ExpertOrchestrator:
        def __init__(self) -> None:
            self.queries: list[str] = []

        async def execute_tasks(self, tasks, context):
            query = context.query
            self.queries.append(query)
            answers = {
                "这个相机修过吗？": "没有维修过。",
                "最低多少？": "最低 ¥1490.00 可以拍。",
            }
            if query == "周日能到吗？":
                return [ExpertResult.handoff(tasks[0], "knowledge_evidence_unavailable")]
            return [ExpertResult.answered(tasks[0], answers[query])]

    async def _load_item(item_id: str) -> dict[str, object]:
        return {"found": True, "item_id": item_id}

    orchestrator = _ExpertOrchestrator()
    handler = XianyuExpertTaskHandler(expert_orchestrator=orchestrator, item_loader=_load_item)  # type: ignore[arg-type]
    tasks = [
        _expert_task("product-1", "product", "这个相机修过吗？", query_target="history.repair_history"),
        _expert_task("price-1", "price", "最低多少？", query_target="price.minimum"),
        _expert_task("service-1", "service", "周日能到吗？", query_target="shipping.dispatch_time"),
    ]
    results = asyncio.run(TaskExecutor({kind: handler for kind in ("product", "price", "service")}).execute(tasks, _message(item_id="ITEM-001"), SessionContext()))

    assert orchestrator.queries == [task.query for task in tasks]
    assert [result.answer for result in results] == [
        "没有维修过。",
        "最低 ¥1490.00 可以拍。",
        "目前只能确认付款后48小时内发出，周日是否能送达暂时无法确认。",
    ]
    assert results[-1].status == "unavailable"


def test_unavailable_delivery_task_replaces_legacy_handoff_wording() -> None:
    class _ExpertOrchestrator:
        async def execute_tasks(self, tasks, context):
            del context
            return [ExpertResult.handoff(tasks[0], "knowledge_evidence_unavailable")]

    async def _load_item(item_id: str) -> dict[str, object]:
        return {"found": True, "item_id": item_id}

    result = asyncio.run(XianyuExpertTaskHandler(expert_orchestrator=_ExpertOrchestrator(), item_loader=_load_item).handle(  # type: ignore[arg-type]
        _expert_task("service-1", "service", "周日能到吗？", query_target="shipping.dispatch_time"),
        _message(),
        SessionContext(),
    ))

    assert result.status == "unavailable"
    assert result.answer == "目前只能确认付款后48小时内发出，周日是否能送达暂时无法确认。"


def test_order_handler_uses_the_standalone_order_route_for_combined_plan() -> None:
    class _OrderHandler:
        def __init__(self) -> None:
            self.order_calls: list[tuple[str, str]] = []

        async def order(self, query: str, order_id: str) -> dict[str, object]:
            self.order_calls.append((query, order_id))
            return {"action": "reply", "answer": "订单正在运输中。"}

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
    assert handler.order_calls == [(task.query, "TEST1001")]


def test_service_handler_uses_planner_execution_mode_not_legacy_combined_route() -> None:
    class _ExpertHandler:
        async def handle(self, *args: object, **kwargs: object) -> TaskResult:
            raise AssertionError("common knowledge must not re-enter the expert route")

    class _KnowledgeResponder:
        async def handle_common(self, query: str, **kwargs: object) -> dict[str, object]:
            assert query == "一般多久发货？"
            assert kwargs == {}
            return {"action": "reply", "can_answer": True, "answer": "付款后 48 小时内发出。"}

    handler = ServiceTaskHandler(
        expert_handler=_ExpertHandler(),  # type: ignore[arg-type]
        knowledge_responder=_KnowledgeResponder(),  # type: ignore[arg-type]
        general_rag=lambda: object(),
    )

    result = asyncio.run(
        handler.handle(
            Task(
                "service-1",
                "service",
                "一般多久发货？",
                execution_mode="common_knowledge",
            ),
            _message(),
            SessionContext(),
        )
    )

    assert result == TaskResult("service-1", "answered", "付款后 48 小时内发出。")
