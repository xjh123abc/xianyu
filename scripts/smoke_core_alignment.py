"""Run the K01-K14 core alignment smoke cases against the local /chat API."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any
from urllib import error, request
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASE_FILE = PROJECT_ROOT / "eval" / "core_alignment_cases.json"


def _post_chat(base_url: str, payload: dict[str, str], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    call = request.Request(
        f"{base_url.rstrip('/')}/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(call, timeout=timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("/chat response must be a JSON object")
        return {"_http_status": response.status, **parsed}


def _run_chat_case(
    case: dict[str, Any],
    *,
    base_url: str,
    timeout: float,
    item_override: str | None,
    run_id: str,
) -> dict[str, Any]:
    chat_id = f"core_{run_id}_{case['id'].lower()}"
    item_id = item_override or case.get("item_id")
    turns: list[dict[str, Any]] = []
    planner_call_count = 0
    for index, turn in enumerate(case.get("turns", []), start=1):
        payload: dict[str, str] = {
            "query": str(turn["query"]),
            "chat_id": chat_id,
        }
        selected_item = turn.get("item_id", item_id)
        if selected_item:
            payload["item_id"] = str(selected_item)
        try:
            response = _post_chat(base_url, payload, timeout)
            http_status = int(response.pop("_http_status", 200))
            successful = 200 <= http_status < 300
            if successful and response.get("reason") != "turn_budget_exhausted":
                # ChatService invokes Planner once for each accepted chat turn.
                planner_call_count += 1
            sources = response.get("sources", [])
            if not isinstance(sources, list):
                sources = []
            turns.append(
                {
                    "turn": index,
                    "request": payload,
                    "http_status": http_status,
                    "answer": response.get("answer"),
                    "sources": sources,
                    "source_count": len(sources),
                    "state": {
                        key: response.get(key)
                        for key in ("chat_id", "item_id", "intent", "action", "turn_id", "proposal_id", "reason")
                        if key in response
                    },
                    "response": response,
                }
            )
        except (error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            turns.append(
                {
                    "turn": index,
                    "request": payload,
                    "http_status": getattr(getattr(exc, "code", None), "__int__", lambda: None)(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    valid_responses = [turn for turn in turns if isinstance(turn.get("http_status"), int)]
    failed = len(valid_responses) != len(turns) or any(
        not 200 <= int(turn["http_status"]) < 300 for turn in valid_responses
    )
    contract_failed = any(
        not turn.get("answer")
        or turn.get("state", {}).get("chat_id") != chat_id
        for turn in turns
        if "response" in turn
    )
    return {
        "case_id": case["id"],
        "status": "failed" if failed or contract_failed else "passed",
        "validation_scope": "HTTP /chat response contract; review field still requires human evaluation",
        "chat_id": chat_id,
        "item_id": item_id,
        "turn_count": len(turns),
        "planner_call_count": planner_call_count,
        "planner_count_basis": "one planner invocation per successful ChatService turn; timeouts before planning are excluded when identified",
        "review": case.get("review", ""),
        "turns": turns,
    }


def run_cases(
    *,
    base_url: str,
    timeout: float,
    case_file: Path,
    item_override: str | None = None,
) -> dict[str, Any]:
    case_data = json.loads(case_file.read_text(encoding="utf-8"))
    if not isinstance(case_data, dict) or not isinstance(case_data.get("cases"), list):
        raise ValueError("case file must contain a cases array")
    run_id = uuid4().hex[:10]
    results: list[dict[str, Any]] = []
    for case in case_data["cases"]:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str):
            raise ValueError("each case must have an id")
        if case.get("execution") != "chat":
            results.append(
                {
                    "case_id": case["id"],
                    "status": "not_run",
                    "review": case.get("review", ""),
                    "turn_count": 0,
                    "planner_call_count": 0,
                    "turns": [],
                }
            )
            continue
        results.append(
            _run_chat_case(
                case,
                base_url=base_url,
                timeout=timeout,
                item_override=item_override,
                run_id=run_id,
            )
        )
    executed = [entry for entry in results if entry["status"] != "not_run"]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_url": base_url,
        "case_file": str(case_file),
        "run_id": run_id,
        "summary": {
            "total_cases": len(results),
            "passed": sum(entry["status"] == "passed" for entry in results),
            "failed": sum(entry["status"] == "failed" for entry in results),
            "not_run": sum(entry["status"] == "not_run" for entry in results),
            "planner_call_count": sum(int(entry["planner_call_count"]) for entry in results),
        },
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=Path("eval/results/local/core_alignment.json"))
    parser.add_argument("--case-file", type=Path, default=CASE_FILE)
    parser.add_argument("--item-id", help="override the TEST fixture item id for chat cases")
    parser.add_argument("--timeout", type=float, default=35.0)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")

    report = run_cases(
        base_url=args.base_url,
        timeout=args.timeout,
        case_file=args.case_file,
        item_override=args.item_id,
    )
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **report["summary"]}, ensure_ascii=False))
    return 1 if report["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
