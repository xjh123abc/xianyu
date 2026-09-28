"""S6 acceptance assets and real-send gate tests."""

from __future__ import annotations

import asyncio
import json
import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
import scripts.run_xianyu_stage3 as stage3_runner

from app.channels.xianyu.acceptance_gate import (
    AcceptanceGateError,
    REQUIRED_ACCEPTANCE_CASES,
    load_s6_acceptance_report,
)
from app.channels.xianyu.models import InboundMessage
from app.services.query_planner import build_expert_plan
from eval.batch_chat_test import QUESTIONS, _result_record, _task_records, _validate_response
from scripts.run_xianyu_stage3 import (
    _account_activation,
    _authorise_start,
    _counts_acceptance_delivery,
    _current_project_commit,
    _log,
    _process_scoped_event,
    _temporary_limit_reached,
)


def _dev_args(**overrides: object) -> Namespace:
    values: dict[str, object] = {
        "dev_live": True,
        "enable": False,
        "controlled_acceptance": False,
        "account": "test-seller",
        "test_chat": "xianyu:seller:test-chat",
        "dev_max_messages": 0,
        "acceptance_report": None,
        "acceptance_query": [],
    }
    values.update(overrides)
    return Namespace(**values)


def _event(message_id: str, chat_id: str, text: str) -> InboundMessage:
    return InboundMessage(
        account_id="test-seller",
        platform_message_id=message_id,
        chat_id=chat_id,
        buyer_id="test-buyer",
        text=text,
        platform_item_id="listing-1",
    )


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
    tasks = _task_records(question, "TEST_CORE_ALIGNMENT_CAMERA")

    record = _result_record(
        "T52",
        "复合问题",
        question,
        "qa_batch_chat_t52",
        "TEST_CORE_ALIGNMENT_CAMERA",
        tasks,
        12.5,
        response={
            "answer": "还在。没有维修过。不包邮最低 ¥1470.00。",
            "action": "reply",
            "can_answer": True,
            "route": "xianyu",
            "reason": None,
            "sources": [{"source": "mcp:get_item_info", "index": "TEST_CORE_ALIGNMENT_CAMERA"}],
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
        xianyu_context={"item_id": "TEST_CORE_ALIGNMENT_CAMERA"},
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
        xianyu_context={"item_id": "TEST_CORE_ALIGNMENT_CAMERA"},
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


def test_dev_live_needs_no_report_query_or_current_commit(monkeypatch) -> None:
    args = _dev_args()
    run_git = Mock(side_effect=AssertionError("dev-live must not inspect project HEAD"))
    monkeypatch.setattr("scripts.run_xianyu_stage3.subprocess.run", run_git)

    assert _current_project_commit(args, Path(".")) is None
    _authorise_start(args, None)

    run_git.assert_not_called()


def test_dev_live_cli_parses_new_options_before_starting(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_run(args: Namespace) -> int:
        captured.update(vars(args))
        return 0

    monkeypatch.setattr(stage3_runner, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_xianyu_stage3.py",
            "--project-root",
            str(tmp_path),
            "--reference-root",
            str(tmp_path / "reference"),
            "--db",
            str(tmp_path / "channel.sqlite3"),
            "--log-file",
            str(tmp_path / "dev-live.log"),
            "--account",
            "test-seller",
            "--dev-live",
            "--test-chat",
            "xianyu:seller:test-chat",
            "--dev-max-messages",
            "3",
        ],
    )

    assert stage3_runner.main() == 0
    assert captured["dev_live"] is True
    assert captured["test_chat"] == "xianyu:seller:test-chat"
    assert captured["dev_max_messages"] == 3
    assert captured["acceptance_report"] is None
    assert captured["acceptance_query"] == []


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"account": ""}, "--account"),
        ({"test_chat": ""}, "--test-chat"),
        ({"enable": True}, "--enable"),
        ({"controlled_acceptance": True}, "--controlled-acceptance"),
        ({"dev_max_messages": -1}, "--dev-max-messages"),
    ],
)
def test_dev_live_rejects_missing_scope_or_conflicting_modes(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _authorise_start(_dev_args(**changes), None)


def test_dev_live_ignores_other_chats_before_calling_worker() -> None:
    args = _dev_args()
    event = _event("other-1", "xianyu:seller:other-chat", "任意问题")
    store = Mock()
    store.record_inbound.return_value = True
    worker = Mock()
    worker.process = AsyncMock()

    result = asyncio.run(_process_scoped_event(args, event, store, worker, Mock()))

    assert result is None
    store.record_inbound.assert_called_once_with(event)
    store.mark_ignored.assert_called_once_with(
        "test-seller", "other-1", "dev_live_scope"
    )
    worker.process.assert_not_awaited()


def test_buyer_event_log_keeps_plain_reason_enum(tmp_path: Path) -> None:
    log_file = tmp_path / "xianyu.log"

    _log(
        log_file,
        "buyer_event",
        "ignored",
        message_hash="message-hash",
        chat_hash="chat-hash",
        delivery=None,
        reason="seller_echo",
        reason_hash="reason-hash",
    )

    record = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert record["result"] == "ignored"
    assert record["reason"] == "seller_echo"
    assert record["reason_hash"] == "reason-hash"


def test_dev_live_processes_multiple_arbitrary_texts_in_the_same_chat() -> None:
    args = _dev_args()
    events = [
        _event("test-1", str(args.test_chat), "最低多少，能便宜点吗？"),
        _event("test-2", str(args.test_chat), "这句话不在任何白名单里"),
    ]
    store = Mock()
    worker = Mock()
    worker.process = AsyncMock(
        side_effect=[
            {"action": "answer", "delivery": "confirmed"},
            {"action": "answer", "delivery": "local_submitted"},
        ]
    )
    sender = Mock()

    results = [
        asyncio.run(_process_scoped_event(args, event, store, worker, sender))
        for event in events
    ]

    assert [result["delivery"] for result in results if result] == [
        "confirmed",
        "local_submitted",
    ]
    assert [call.args[0].chat_id for call in worker.process.await_args_list] == [
        args.test_chat,
        args.test_chat,
    ]
    assert [call.args[0].text for call in worker.process.await_args_list] == [
        "最低多少，能便宜点吗？",
        "这句话不在任何白名单里",
    ]
    store.record_inbound.assert_not_called()


def test_dev_live_max_messages_zero_is_unbounded_and_positive_stops_at_limit() -> None:
    unlimited = _dev_args(dev_max_messages=0)
    limited = _dev_args(dev_max_messages=2)

    assert _temporary_limit_reached(unlimited, 1000) is False
    assert _temporary_limit_reached(limited, 1) is False
    assert _temporary_limit_reached(limited, 2) is True


@pytest.mark.parametrize("failure", [RuntimeError("boom"), KeyboardInterrupt()])
def test_dev_live_always_pauses_after_error_or_ctrl_c(failure: BaseException) -> None:
    worker = Mock()

    with pytest.raises(type(failure)):
        with _account_activation(_dev_args(), worker):
            raise failure

    worker.enable_account.assert_called_once_with()
    worker.pause_account.assert_called_once_with()


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
