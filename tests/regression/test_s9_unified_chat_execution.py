"""S9 acceptance coverage for the single ChatService execution path."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from app.services.chat_contracts import ChatMessage, SessionContext, Task, TaskResult
from app.services.chat_service import ChatService
from app.services.item_service import ItemService
from app.services.session_manager import SessionManager


_QUERY = "这个相机修过吗？最低多少？周日能到吗？"
_TASKS = [
    Task("product-1", "product", "这个相机修过吗？", {"_planner_state": {
        "intent_match": {"intent": "PRODUCT", "required_fields": [], "source": "rule"},
        "question_plan": {"item_fields": [], "knowledge_questions": []},
        "route": "rag",
        "order_id": None,
        "needs_item": True,
        "defer_to_order_context": False,
        "is_rule_followup": False,
        "use_xianyu_without_item": True,
    }}),
    Task("price-1", "price", "最低多少？", {"_planner_state": {
        "intent_match": {"intent": "PRODUCT", "required_fields": [], "source": "rule"},
        "question_plan": {"item_fields": [], "knowledge_questions": []},
        "route": "rag",
        "order_id": None,
        "needs_item": True,
        "defer_to_order_context": False,
        "is_rule_followup": False,
        "use_xianyu_without_item": True,
    }}),
    Task("service-1", "service", "周日能到吗？", {"_planner_state": {
        "intent_match": {"intent": "PRODUCT", "required_fields": [], "source": "rule"},
        "question_plan": {"item_fields": [], "knowledge_questions": []},
        "route": "rag",
        "order_id": None,
        "needs_item": True,
        "defer_to_order_context": False,
        "is_rule_followup": False,
        "use_xianyu_without_item": True,
    }}),
]


class _Planner:
    def plan(self, query: str, context: SessionContext, *, item_id: str | None = None) -> list[Task]:
        assert query == _QUERY
        assert item_id == "CANON_FTB_001"
        del context
        return _TASKS

    @staticmethod
    def xianyu_context_updates(query: str, context: object) -> dict[str, str]:
        del query, context
        return {}


class _Executor:
    def __init__(self) -> None:
        self.tasks: list[Task] = []

    async def execute(
        self,
        tasks: Sequence[Task],
        message: ChatMessage,
        context: SessionContext,
    ) -> list[TaskResult]:
        self.tasks = list(tasks)
        assert message.item_id == "CANON_FTB_001"
        assert context.current_item_id == "CANON_FTB_001"
        return [
            TaskResult("product-1", "answered", "没有维修记录。"),
            TaskResult("price-1", "answered", "最低 100 元。"),
            TaskResult("service-1", "answered", "周日可送达。"),
        ]


class _Merger:
    def __init__(self) -> None:
        self.tasks: list[Task] = []
        self.results: list[TaskResult] = []

    def merge(self, query: str, tasks: Sequence[Task], results: Sequence[TaskResult]) -> dict[str, object]:
        assert query == _QUERY
        self.tasks = list(tasks)
        self.results = list(results)
        return {"action": "reply", "answer": "已合并三个任务结果。", "can_answer": True}


class _Mcp:
    async def get_item_info(self, item_id: str) -> dict[str, object]:
        assert item_id == "CANON_FTB_001"
        return ItemService().get_item_info(item_id)


def test_chat_service_sends_product_price_and_service_results_to_result_merger() -> None:
    executor = _Executor()
    service = ChatService(
        mcp_service=_Mcp(),  # type: ignore[arg-type]
        planner=_Planner(),  # type: ignore[arg-type]
        task_executor=executor,  # type: ignore[arg-type]
        session_manager=SessionManager(),
    )
    merger = _Merger()
    service.result_merger = merger  # type: ignore[assignment]

    response = asyncio.run(service.chat_async(_QUERY, "s9-unified", item_id="CANON_FTB_001"))

    assert response["action"] == "reply"
    assert [task.task_type for task in executor.tasks] == ["product", "price", "service"]
    assert merger.tasks == executor.tasks
    assert [result.task_id for result in merger.results] == [
        "product-1",
        "price-1",
        "service-1",
    ]
