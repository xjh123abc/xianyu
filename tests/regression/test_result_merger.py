"""S6 checks for unified task-result merging."""

from app.services.chat_contracts import Task, TaskResult
from app.services.result_merger import ResultMerger


def test_merger_preserves_a_single_legacy_handler_response() -> None:
    response = {"route": "order", "answer": "订单已发货", "can_answer": True}

    merged = ResultMerger().merge(
        "查订单",
        [Task("order-1", "order", "查订单")],
        [
            TaskResult(
                "order-1",
                "answered",
                "订单已发货",
                metadata={"response": response},
            )
        ],
    )

    assert merged == {**response, "query": "查订单"}


def test_merger_combines_answered_tasks_in_planner_order() -> None:
    tasks = [
        Task("order-1", "order", "订单在哪"),
        Task("service-1", "service", "多久到"),
    ]

    merged = ResultMerger().merge(
        "订单在哪，多久到",
        tasks,
        [
            TaskResult("order-1", "answered", "订单正在运输中", [{"source": "mcp", "index": 1}]),
            TaskResult("service-1", "answered", "一般 24 小时内发货", [{"source": "rules", "index": 2}]),
        ],
    )

    assert merged["answer"] == "订单正在运输中\n一般 24 小时内发货"
    assert merged["task_types"] == ["order", "service"]
    assert merged["sources"] == [{"source": "mcp", "index": 1}, {"source": "rules", "index": 2}]
