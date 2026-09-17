"""Receive-only probe for an authorized Xianyu WebSocket session.

This script deliberately imports only the transport helpers from a pinned
reference checkout.  It never imports the reference project's Agent,
context manager, or automatic reply loop, and it has no send operation.

The reference checkout is intentionally kept outside this repository.  Run
with a disposable environment containing its small transport dependencies.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.channels.xianyu.reference_runtime import (
    DEFAULT_REFERENCE_COMMIT,
    DEFAULT_WS_URL,
    _decode_payload,
    _install_reference_imports,
    _load_cookie,
    _registration_messages,
    _set_request_timeout,
)


DEFAULT_TIMEOUT_SECONDS = 30.0


def _hash_identifier(value: object) -> str | None:
    """Return a short stable digest without exposing platform identifiers."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _event_summary(event: Mapping[str, Any], my_id: str, decrypt: Any) -> dict[str, Any]:
    """Summarize one decoded event using hashes for identifiers and text."""

    first = event.get("1")
    details = first.get("10") if isinstance(first, Mapping) else None
    if not isinstance(first, Mapping) or not isinstance(details, Mapping):
        return {"kind": "non_chat", "top_level_keys": sorted(str(key) for key in event)}

    sender_id = details.get("senderUserId")
    content = details.get("reminderContent")
    reminder_url = str(details.get("reminderUrl") or "")
    item_id = None
    if "itemId=" in reminder_url:
        item_id = reminder_url.split("itemId=", 1)[1].split("&", 1)[0]
    kind = "seller_chat" if str(sender_id) == my_id else "buyer_chat"
    if not isinstance(content, str) or not content.strip():
        kind = "non_text"
    return {
        "kind": kind,
        "chat_id_hash": _hash_identifier(first.get("2")),
        "sender_id_hash": _hash_identifier(sender_id),
        "item_id_hash": _hash_identifier(item_id),
        "created_at": first.get("5"),
        "text_present": isinstance(content, str) and bool(content.strip()),
        "text_length": len(content) if isinstance(content, str) else 0,
        "text_hash": _hash_identifier(content),
        "content_type": details.get("contentType"),
    }


def _record_message(
    message: Mapping[str, Any],
    my_id: str,
    decrypt: Any,
    report: dict[str, Any],
) -> None:
    """Classify all events in one WebSocket message."""

    body = message.get("body")
    package = body.get("syncPushPackage") if isinstance(body, Mapping) else None
    records = package.get("data") if isinstance(package, Mapping) else None
    if not isinstance(records, list):
        if message.get("code") == 200:
            report["heartbeat_responses"] += 1
        else:
            report["other_messages"] += 1
        return

    report["sync_packages"] += 1
    for record in records:
        raw_data = record.get("data") if isinstance(record, Mapping) else None
        event = _decode_payload(raw_data, decrypt)
        if event is None:
            report["decode_failures"] += 1
            continue
        report["decoded_events"] += 1
        summary = _event_summary(event, my_id, decrypt)
        kind = summary.get("kind")
        if kind == "buyer_chat":
            report["buyer_messages"] += 1
        elif kind == "seller_chat":
            report["seller_messages"] += 1
        else:
            report["other_events"] += 1
        if len(report["events"]) < 20:
            report["events"].append(summary)


async def _receive(
    cookie_header: str,
    parsed_cookie: Mapping[str, str],
    timeout: float,
    reference_root: Path,
) -> dict[str, Any]:
    """Validate login, register the socket, and receive without sending replies."""

    _install_reference_imports(reference_root)
    from XianyuApis import XianyuApis
    from utils.xianyu_utils import decrypt, generate_device_id, generate_mid

    api = XianyuApis()
    _set_request_timeout(api.session, timeout=min(timeout, 15.0))
    api.session.cookies.update(dict(parsed_cookie))
    report: dict[str, Any] = {
        "login_ok": False,
        "token_ok": False,
        "websocket_connected": False,
        "heartbeat_responses": 0,
        "sync_packages": 0,
        "decoded_events": 0,
        "buyer_messages": 0,
        "seller_messages": 0,
        "other_events": 0,
        "decode_failures": 0,
        "other_messages": 0,
        "events": [],
    }

    if not api.hasLogin():
        report["status"] = "login_failed"
        return report
    report["login_ok"] = True
    my_id = str(api.session.cookies.get("unb") or "")
    if not my_id:
        report["status"] = "login_missing_account_id"
        return report

    device_id = generate_device_id(my_id)
    try:
        token_result = api.get_token(device_id)
    except SystemExit:
        report["status"] = "token_failed"
        report["token_error_type"] = "SystemExit"
        return report
    except Exception as exc:
        report["status"] = "token_failed"
        report["token_error_type"] = type(exc).__name__
        return report
    token = token_result.get("data", {}).get("accessToken") if isinstance(token_result, Mapping) else None
    if not isinstance(token, str) or not token.strip():
        report["status"] = "token_missing"
        return report
    report["token_ok"] = True

    import websockets

    refreshed_cookie = "; ".join(
        f"{name}={value}" for name, value in api.session.cookies.get_dict().items()
    ) or cookie_header
    headers = {
        "Cookie": refreshed_cookie,
        "Host": "wss-goofish.dingtalk.com",
        "Origin": "https://www.goofish.com",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/133.0.0.0 Safari/537.36",
    }
    connect_kwargs: dict[str, Any] = {"open_timeout": min(timeout, 15.0)}
    parameter_names = inspect.signature(websockets.connect).parameters
    header_key = "additional_headers" if "additional_headers" in parameter_names else "extra_headers"
    connect_kwargs[header_key] = headers
    registration, acknowledgement = _registration_messages(token, device_id, generate_mid)

    try:
        async with websockets.connect(DEFAULT_WS_URL, **connect_kwargs) as websocket:
            report["websocket_connected"] = True
            await websocket.send(registration)
            await asyncio.sleep(1)
            await websocket.send(acknowledgement)
            deadline = asyncio.get_running_loop().time() + timeout
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    raw_message = await asyncio.wait_for(websocket.recv(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                if isinstance(raw_message, bytes):
                    raw_message = raw_message.decode("utf-8", errors="replace")
                try:
                    message = json.loads(raw_message)
                except (TypeError, json.JSONDecodeError):
                    report["other_messages"] += 1
                    continue
                if isinstance(message, Mapping):
                    _record_message(message, my_id, decrypt, report)
    except Exception as exc:
        report["status"] = "websocket_error"
        report["error_type"] = type(exc).__name__
        return report

    report["status"] = "received_buyer_message" if report["buyer_messages"] else "connected_no_buyer_message"
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--reference-commit", default=DEFAULT_REFERENCE_COMMIT)
    parser.add_argument("--receive-timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.receive_timeout <= 0:
        raise SystemExit("--receive-timeout must be greater than zero")
    try:
        project_root = args.project_root.resolve()
        reference_root = args.reference_root.resolve()
        cookie_header, parsed_cookie, cookie_metadata = _load_cookie(project_root)
        if args.reference_commit != DEFAULT_REFERENCE_COMMIT:
            raise ValueError("reference checkout commit does not match the pinned stage-one commit")
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory(prefix="xianyu-probe-") as isolated_cwd:
            try:
                os.chdir(isolated_cwd)
                result = asyncio.run(
                    _receive(
                        cookie_header,
                        parsed_cookie,
                        args.receive_timeout,
                        reference_root,
                    )
                )
            finally:
                # Restore before TemporaryDirectory attempts Windows cleanup.
                os.chdir(original_cwd)
        result["cookie"] = cookie_metadata
        result["reference_commit"] = args.reference_commit
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result.get("status") in {"received_buyer_message", "connected_no_buyer_message"} else 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "probe_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
