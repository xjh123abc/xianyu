"""Unit checks for the live acceptance runner itself (no live services)."""

from __future__ import annotations

from scripts.smoke_unified_chat import _record, run_acceptance


def _fake_chat(payload: dict[str, str]) -> dict[str, object]:
    query = payload["query"]
    item_id = payload.get("item_id")
    chat_id = payload["chat_id"]

    if chat_id == "qa_a05_missing_item" or chat_id == "qa_a08_fresh":
        return {
            "action": "clarify",
            "answer": "请补充商品编号或具体商品信息。",
            "item_id": None,
            "can_answer": False,
            "reason": "buyer_question_requires_clarification",
        }
    if item_id == "XXX999":
        return {
            "action": "clarify",
            "item_id": "XXX999",
            "item_info": {"found": False},
            "answer": "这件商品我暂时没查到，麻烦确认一下。",
            "can_answer": False,
            "reason": "item_context_conflict",
        }

    remembered = "DEMO_ITEM_002" if chat_id == "qa_a08_switch" and item_id is None else item_id
    if chat_id == "qa_a07_memory" and item_id is None:
        remembered = "DEMO_ITEM_001"
    sources: list[dict[str, object]] = []
    answer = ""
    can_answer = True
    if remembered:
        sources.append({"source": "mcp:get_item_info", "index": remembered})
    if "多少钱" in query or "标价" in query:
        answer += "标价为 1280.00 元。"
    if "还能买" in query or "在售" in query:
        if remembered == "DEMO_ITEM_002":
            answer += "这件已经出掉了。"
        elif remembered == "DEMO_ITEM_003":
            return {
                "action": "reply",
                "answer": "该问题目前暂无足够信息确认。",
                "item_id": remembered,
                "item_info": {"found": True, "sale_status": "unknown"},
                "can_answer": False,
                "reason": "sale_status_unavailable",
                "sources": sources,
                "results": [],
            }
    if any(term in query for term in ("售后", "质量问题", "退货")):
        sources.append({"source": "common/seller_rules.md", "index": 0})
        answer += "按卖家已确认规则处理。"
    if any(term in query for term in ("配件", "摔过", "USB")):
        sources.append({"source": "items/DEMO_ITEM_001.md", "index": 0})
        answer += "请以商品资料为准。"
    return {
        "action": "reply",
        "answer": answer,
        "item_id": remembered,
        "item_info": {
            "found": True,
            "sale_status": "sold" if remembered == "DEMO_ITEM_002" else "listed",
            "listed_price_cents": 128000 if remembered == "DEMO_ITEM_001" else None,
            "facts": {"history": {"drop_history": "unknown"}},
        }
        if remembered
        else None,
        "can_answer": can_answer,
        "sources": sources,
        "results": sources,
    }


def test_runner_records_every_acceptance_case_without_mode_parameters() -> None:
    records = run_acceptance(_fake_chat)

    assert [record["case_id"] for record in records] == [f"A{index:02}" for index in range(1, 15)]
    assert all(
        "scene" not in payload and "scenario" not in payload
        for record in records
        for payload in record["input"]
    )
    assert all(payload["chat_id"].startswith("qa_") for record in records for payload in record["input"])
    assert not [record for record in records if record["status"] == "失败"]
    assert sum(record["status"] == "通过" for record in records) == 5
    assert sum(record["status"] == "阻塞" for record in records) == 9


def test_open_ended_answer_is_not_automatically_marked_passed() -> None:
    record = _record(
        "A02",
        [{"query": "售后？", "chat_id": "qa_test"}],
        "符合真实规则",
        [{"answer": "售后"}],
        [(True, "来源边界正确")],
        manual_review="与原文逐项核对",
    )

    assert record["status"] == "阻塞"
    assert "人工" in record["reason"]


def test_failed_boundary_check_is_reported_as_failure() -> None:
    record = _record(
        "A09",
        [],
        "不得跨商品",
        [],
        [(False, "错误引入 002")],
        manual_review="检查答案措辞",
    )

    assert record["status"] == "失败"
    assert record["reason"] == "错误引入 002"


def test_live_request_error_is_recorded_as_blocked() -> None:
    record = _record(
        "A03",
        [{"query": "组合问题", "chat_id": "qa_error"}],
        "真实链路回答",
        [{"_request_error": "connection reset"}],
        [(False, "没有响应")],
    )

    assert record["status"] == "阻塞"
    assert "connection reset" in record["reason"]
