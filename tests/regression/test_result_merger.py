"""S6 checks for unified task-result merging."""

from app.services.chat_contracts import Task, TaskResult
from app.services.result_merger import ResultMerger
import pytest


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


@pytest.mark.parametrize(
    "statuses",
    [
        ["answered", "answered", "answered"],
        ["answered", "answered", "unavailable"],
        ["answered", "unavailable", "unavailable"],
        ["unavailable", "unavailable", "unavailable"],
    ],
    ids=["all_answered", "two_answered", "one_answered", "none_answered"],
)
def test_merger_keeps_partial_success_without_human_handoff(statuses: list[str]) -> None:
    tasks = [
        Task("product-1", "product", "商品修过吗"),
        Task("price-1", "price", "最低多少"),
        Task("service-1", "service", "周日是否发货"),
    ]
    results = [
        TaskResult(task.task_id, status, f"{task.task_type} 已确认" if status == "answered" else "", reason="unavailable")
        for task, status in zip(tasks, statuses)
    ]

    merged = ResultMerger().merge("组合问题", tasks, results)

    assert merged["action"] in {"reply", "clarify"}
    assert merged["action"] != "handoff"
    assert "human_handoff" not in str(merged)
    if "unavailable" in statuses:
        assert "该问题目前暂无足够信息确认。" not in merged["answer"]
    if statuses.count("answered") == 2:
        assert "product 已确认" in merged["answer"]
        assert "price 已确认" in merged["answer"]


def test_merger_includes_every_result_when_service_is_unavailable() -> None:
    tasks = [
        Task("product-1", "product", "修过吗"),
        Task("price-1", "price", "最低多少"),
        Task("service-1", "service", "多久发货"),
    ]
    results = [
        TaskResult("product-1", "answered", "商品没有维修记录。"),
        TaskResult("price-1", "answered", "目前价格可以小刀。"),
        TaskResult("service-1", "unavailable", "", reason="knowledge_evidence_unavailable"),
    ]

    merged = ResultMerger().merge("组合问题", tasks, results)

    assert merged["action"] == "reply"
    assert "商品没有维修记录。" in merged["answer"]
    assert "目前价格可以小刀。" in merged["answer"]
    assert "该问题目前暂无足够信息确认。" not in merged["answer"]
