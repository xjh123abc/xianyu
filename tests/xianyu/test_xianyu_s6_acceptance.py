"""S6 acceptance assets and real-send gate tests."""

from __future__ import annotations

import json
from argparse import Namespace

import pytest

from app.channels.xianyu.acceptance_gate import (
    AcceptanceGateError,
    REQUIRED_ACCEPTANCE_CASES,
    load_s6_acceptance_report,
)
from app.services.query_planner import build_expert_plan
from eval.batch_chat_test import QUESTIONS, _result_record, _task_records, _validate_response
from scripts.run_xianyu_stage3 import _authorise_start, _counts_acceptance_delivery


def _passed_report(commit: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "git_commit": commit,
        "approved_for_auto_send": True,
        "phases": {
            "code_tests": {"status": "passed", "passed": 1, "failed": 0},
            "fixed_batch": {"status": "passed", "total": 60, "failed": 0},
            "real_model": {"status": "passed"},
            "real_channel": {"status": "passed"},
        },
        "requirements": {case: "passed" for case in REQUIRED_ACCEPTANCE_CASES},
    }


def test_fixed_batch_contains_60_unique_isolated_questions() -> None:
    ids = [case[0] for case in QUESTIONS]

    assert len(QUESTIONS) == 60
    assert len(set(ids)) == 60
    assert ids == [f"T{index:02}" for index in range(1, 61)]
    assert all(question.strip() for _, _, question in QUESTIONS)


def test_fixed_batch_records_tasks_reason_sources_and_elapsed_time() -> None:
    question = "还在吗有没有维修过不包邮最低多少"
    tasks = _task_records(question, "CANON_FTB_001")

    record = _result_record(
        "T52",
        "复合问题",
        question,
        "qa_batch_chat_t52",
        "CANON_FTB_001",
        tasks,
        12.5,
        response={
            "answer": "还在。没有维修过。不包邮最低 ¥1470.00。",
            "action": "reply",
            "can_answer": True,
            "route": "xianyu",
            "reason": None,
            "sources": [{"source": "mcp:get_item_info", "index": "CANON_FTB_001"}],
        },
    )

    assert [task["expert"] for task in record["tasks"]] == [
        "product",
        "product",
        "price",
    ]
    assert all(task["original_question"] for task in record["tasks"])
    assert all(task["normalized_question"] for task in record["tasks"])
    assert all(task["query_target"] for task in record["tasks"])
    assert record["reason"] is None
    assert record["sources"]
    assert record["elapsed_ms"] == 12.5


@pytest.mark.parametrize(
    "response",
    [
        {"route": "rag", "action": "reply", "answer": "越界", "can_answer": True},
        {"route": "xianyu", "action": "reply", "answer": "", "can_answer": True},
        {
            "route": "xianyu",
            "action": "handoff",
            "answer": "需要卖家确认",
            "can_answer": False,
            "reason": "missing",
        },
        {
            "route": "xianyu",
            "action": "handoff",
            "answer": "稍等我看看",
            "can_answer": False,
        },
    ],
)
def test_fixed_batch_rejects_unsafe_or_contradictory_contracts(response) -> None:
    with pytest.raises(RuntimeError):
        _validate_response(response)


def test_model_planner_cannot_turn_current_item_shipping_into_seller_rule_rag() -> None:
    def planner(*args, **kwargs):
        return {
            "tasks": [
                {
                    "task_id": "model1",
                    "expert": "service",
                    "question_fragment": "今天能发吗？",
                    "normalized_question": "今天能否发货",
                    "knowledge_scope": "seller_rule",
                    "transaction_conditions": {},
                    "depends_on_task_ids": [],
                }
            ]
        }

    tasks = build_expert_plan(
        "今天能发吗？走顺丰吗？",
        planner=planner,
        xianyu_context={"item_id": "CANON_FTB_001"},
    )

    assert [(task.normalized_question, task.knowledge_scope) for task in tasks] == [
        ("发货时限或地点", "item_fact"),
        ("快递方式", "item_fact"),
    ]


def test_model_planner_cannot_invent_a_greeting_task() -> None:
    def planner(*args, **kwargs):
        return {
            "tasks": [
                {
                    "task_id": "model-greeting",
                    "expert": "service",
                    "question_fragment": "还在吗",
                    "normalized_question": "普通招呼",
                    "knowledge_scope": "greeting",
                    "transaction_conditions": {},
                    "depends_on_task_ids": [],
                }
            ]
        }

    tasks = build_expert_plan(
        "还在吗有没有维修过不包邮最低多少",
        planner=planner,
        xianyu_context={"item_id": "CANON_FTB_001"},
    )

    assert all(task.knowledge_scope != "greeting" for task in tasks)


def test_s6_gate_accepts_only_a_complete_report_for_the_current_commit(tmp_path) -> None:
    report_path = tmp_path / "s6.json"
    report_path.write_text(json.dumps(_passed_report("abc123")), encoding="utf-8")

    loaded = load_s6_acceptance_report(report_path, expected_commit="abc123")

    assert loaded["approved_for_auto_send"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda report: report.update({"git_commit": "old"}), "Git commit"),
        (
            lambda report: report["phases"]["real_model"].update({"status": "not_run"}),
            "real_model",
        ),
        (
            lambda report: report["phases"]["real_channel"].update({"status": "not_run"}),
            "real_channel",
        ),
        (
            lambda report: report["phases"]["fixed_batch"].update({"total": 59}),
            "60 results",
        ),
        (lambda report: report["requirements"].update({"A10": "failed"}), "A10"),
        (
            lambda report: report.update({"approved_for_auto_send": False}),
            "not approved",
        ),
    ],
)
def test_s6_gate_rejects_incomplete_or_stale_reports(
    tmp_path,
    mutation,
    message: str,
) -> None:
    report = _passed_report("abc123")
    mutation(report)
    report_path = tmp_path / "s6.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(AcceptanceGateError, match=message):
        load_s6_acceptance_report(report_path, expected_commit="abc123")


def test_real_worker_requires_report_outside_controlled_acceptance() -> None:
    args = Namespace(
        controlled_acceptance=False,
        acceptance_report=None,
    )

    with pytest.raises(ValueError, match="acceptance-report"):
        _authorise_start(args, "abc123")


def test_controlled_acceptance_requires_one_exact_account_chat_and_query() -> None:
    valid = Namespace(
        controlled_acceptance=True,
        enable=False,
        account="test-seller",
        acceptance_chat="xianyu:test:chat",
        acceptance_query=["包邮最低多少？"],
        acceptance_max_messages=1,
        acceptance_report=None,
    )

    _authorise_start(valid, "abc123")

    for field in ("account", "acceptance_chat"):
        invalid = Namespace(**vars(valid))
        setattr(invalid, field, "")
        with pytest.raises(ValueError, match=field.replace("_", "-")):
            _authorise_start(invalid, "abc123")


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ({"action": "answer", "delivery": "confirmed"}, True),
        ({"action": "human_handoff", "delivery": "local_submitted"}, True),
        ({"action": "duplicate"}, False),
        ({"action": "answer", "delivery": "failed"}, False),
        ({"action": "superseded", "delivery": "confirmed"}, False),
    ],
)
def test_controlled_acceptance_counts_only_buyer_visible_submissions(
    result,
    expected: bool,
) -> None:
    assert _counts_acceptance_delivery(result) is expected
