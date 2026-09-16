"""Fail-closed S6 gate for enabling real Xianyu automatic delivery."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


REQUIRED_ACCEPTANCE_CASES = tuple(f"A{index:02}" for index in range(1, 19))
REQUIRED_PHASES = (
    "code_tests",
    "fixed_batch",
    "real_model",
    "real_channel",
)


class AcceptanceGateError(ValueError):
    """Raised when a report cannot authorise real automatic sending."""


def load_s6_acceptance_report(
    path: str | Path,
    *,
    expected_commit: str,
) -> dict[str, Any]:
    """Load and validate one report bound to the exact code being enabled."""

    report_path = Path(path)
    try:
        raw = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AcceptanceGateError("S6 acceptance report is unreadable or invalid") from exc
    if not isinstance(raw, Mapping):
        raise AcceptanceGateError("S6 acceptance report must be a JSON object")
    report = dict(raw)
    _validate_s6_acceptance_report(report, expected_commit=expected_commit)
    return report


def _validate_s6_acceptance_report(
    report: Mapping[str, Any],
    *,
    expected_commit: str,
) -> None:
    if report.get("schema_version") != 1:
        raise AcceptanceGateError("unsupported S6 acceptance report schema")
    if report.get("git_commit") != expected_commit:
        raise AcceptanceGateError("S6 acceptance report does not match current Git commit")
    if report.get("approved_for_auto_send") is not True:
        raise AcceptanceGateError("S6 acceptance report has not approved automatic sending")

    phases = report.get("phases")
    if not isinstance(phases, Mapping):
        raise AcceptanceGateError("S6 acceptance phases are missing")
    for phase in REQUIRED_PHASES:
        value = phases.get(phase)
        if not isinstance(value, Mapping) or value.get("status") != "passed":
            raise AcceptanceGateError(f"S6 phase has not passed: {phase}")

    fixed_batch = phases["fixed_batch"]
    if fixed_batch.get("total") != 60 or fixed_batch.get("failed") != 0:
        raise AcceptanceGateError("S6 fixed batch must contain 60 results and zero failures")

    requirements = report.get("requirements")
    if not isinstance(requirements, Mapping):
        raise AcceptanceGateError("S6 A01-A18 results are missing")
    missing = [case for case in REQUIRED_ACCEPTANCE_CASES if requirements.get(case) != "passed"]
    if missing:
        raise AcceptanceGateError("S6 acceptance cases have not passed: " + ",".join(missing))
