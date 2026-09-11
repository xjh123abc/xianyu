"""One-shot real-channel acceptance test for the pinned Xianyu template.

The process validates the configured browser cookie, obtains a channel token,
registers the WebSocket, waits for one exact buyer trigger, and submits exactly
one fixed reply.  It never imports the template Agent or calls this project's
RAG, MCP, or /chat endpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import builtins
import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from probe_xianyu_channel import (
    DEFAULT_REFERENCE_COMMIT,
    DEFAULT_WS_URL,
    _decode_payload,
    _install_reference_imports,
    _load_cookie,
    _registration_messages,
    _set_request_timeout,
)


FIXED_REPLY = "闲鱼AI客服接通测试成功"
DEFAULT_TRIGGER = "闲鱼渠道接通测试"
_LOG_FILE: Path | None = None


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _digest(value: object) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _log(stage: str, result: str, **details: object) -> None:
    record = {"time": _now(), "stage": stage, "result": result, **details}
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    print(line, flush=True)
    if _LOG_FILE is not None:
        with _LOG_FILE.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def _verify_reference(reference_root: Path) -> None:
    completed = subprocess.run(
        ["git", "-C", str(reference_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    actual = completed.stdout.strip()
    if actual != DEFAULT_REFERENCE_COMMIT:
        raise ValueError(
            f"template commit mismatch: expected {DEFAULT_REFERENCE_COMMIT}, got {actual}"
        )


def _extract_chat_event(event: Mapping[str, Any]) -> dict[str, Any] | None:
    first = event.get("1")
    details = first.get("10") if isinstance(first, Mapping) else None
    if not isinstance(first, Mapping) or not isinstance(details, Mapping):
        return None
    chat_value = str(first.get("2") or "")
    chat_id = chat_value.split("@", 1)[0]
    sender_id = str(details.get("senderUserId") or "")
    text = details.get("reminderContent")
    if not chat_id or not sender_id or not isinstance(text, str):
        return None
    return {
        "chat_id": chat_id,
        "sender_id": sender_id,
        "text": text,
        "created_at": first.get("5"),
    }


def _send_message_payload(
    *, chat_id: str, receiver_id: str, seller_id: str, text: str, generate_mid: Any,
    generate_uuid: Any,
) -> tuple[str, str]:
    inner = {"contentType": 1, "text": {"text": text}}
    encoded = base64.b64encode(
        json.dumps(inner, ensure_ascii=False).encode("utf-8")
    ).decode("ascii")
    send_mid = str(generate_mid())
    payload = {
        "lwp": "/r/MessageSend/sendByReceiverScope",
        "headers": {"mid": send_mid},
        "body": [
            {
                "uuid": generate_uuid(),
                "cid": f"{chat_id}@goofish",
                "conversationType": 1,
                "content": {
                    "contentType": 101,
                    "custom": {"type": 1, "data": encoded},
                },
                "redPointPolicy": 0,
                "extension": {"extJson": "{}"},
                "ctx": {"appVersion": "1.0", "platform": "web"},
                "mtags": {},
                "msgReadStatusSetting": 1,
            },
            {
                "actualReceivers": [
                    f"{receiver_id}@goofish",
                    f"{seller_id}@goofish",
                ]
            },
        ],
    }
    return send_mid, json.dumps(payload, ensure_ascii=False)


async def _heartbeat(websocket: Any, generate_mid: Any, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=15)
        except asyncio.TimeoutError:
            heartbeat_mid = str(generate_mid())
            await websocket.send(json.dumps({"lwp": "/!", "headers": {"mid": heartbeat_mid}}))
            _log("heartbeat", "submitted", request_hash=_digest(heartbeat_mid))


async def _run(
    *,
    project_root: Path,
    reference_root: Path,
    trigger_text: str,
    listen_timeout: float,
    receipt_timeout: float,
    expected_count: int,
) -> dict[str, Any]:
    _verify_reference(reference_root)
    _log("template", "verified", commit=DEFAULT_REFERENCE_COMMIT)

    cookie_header, parsed_cookie, cookie_metadata = _load_cookie(project_root)
    _log("cookie", "validated", **cookie_metadata)

    _install_reference_imports(reference_root)
    from XianyuApis import XianyuApis
    from utils.xianyu_utils import decrypt, generate_device_id, generate_mid, generate_uuid

    api = XianyuApis()
    _set_request_timeout(api.session, timeout=min(listen_timeout, 15.0))
    api.session.cookies.update(dict(parsed_cookie))

    original_input = builtins.input
    builtins.input = lambda _prompt="": ""
    try:
        if not api.hasLogin():
            _log("login", "failed")
            return {"status": "login_failed", "reply_submitted": False}
        _log("login", "passed")

        seller_id = str(api.session.cookies.get("unb") or "")
        if not seller_id:
            _log("login", "failed", reason="missing_seller_id")
            return {"status": "login_missing_account_id", "reply_submitted": False}

        device_id = generate_device_id(seller_id)
        try:
            token_result = api.get_token(device_id)
        except SystemExit:
            _log("token", "failed", reason="template_requested_relogin")
            return {"status": "token_failed", "reply_submitted": False}
    finally:
        builtins.input = original_input

    token = token_result.get("data", {}).get("accessToken") if isinstance(token_result, Mapping) else None
    if not isinstance(token, str) or not token.strip():
        _log("token", "failed", reason="missing_access_token")
        return {"status": "token_missing", "reply_submitted": False}
    _log("token", "passed")

    import websockets

    refreshed_cookie = "; ".join(
        f"{name}={value}" for name, value in api.session.cookies.get_dict().items()
    ) or cookie_header
    headers = {
        "Cookie": refreshed_cookie,
        "Host": "wss-goofish.dingtalk.com",
        "Origin": "https://www.goofish.com",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/133.0.0.0 Safari/537.36"
        ),
    }
    connect_kwargs: dict[str, Any] = {"open_timeout": min(listen_timeout, 15.0)}
    parameter_names = inspect.signature(websockets.connect).parameters
    header_key = "additional_headers" if "additional_headers" in parameter_names else "extra_headers"
    connect_kwargs[header_key] = headers
    registration, acknowledgement = _registration_messages(token, device_id, generate_mid)

    report: dict[str, Any] = {
        "status": "connected_no_trigger",
        "buyer_messages": 0,
        "ignored_buyer_messages": 0,
        "reply_submissions": 0,
        "send_acks": 0,
        "self_echo_count": 0,
    }
    stop_heartbeat = asyncio.Event()
    heartbeat_task: asyncio.Task[None] | None = None
    send_mid: str | None = None
    send_deadline: float | None = None
    pending_send_mids: set[str] = set()

    try:
        async with websockets.connect(DEFAULT_WS_URL, **connect_kwargs) as websocket:
            _log("websocket", "connected", url=DEFAULT_WS_URL)
            await websocket.send(registration)
            _log("websocket_registration", "submitted")
            await asyncio.sleep(1)
            await websocket.send(acknowledgement)
            _log("sync_registration", "submitted")
            heartbeat_task = asyncio.create_task(_heartbeat(websocket, generate_mid, stop_heartbeat))

            loop = asyncio.get_running_loop()
            listen_deadline = loop.time() + listen_timeout
            while True:
                active_deadline = send_deadline if send_deadline is not None else listen_deadline
                remaining = active_deadline - loop.time()
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
                    _log("websocket_message", "ignored_invalid_json")
                    continue
                if not isinstance(message, Mapping):
                    continue

                message_headers = message.get("headers")
                incoming_mid = message_headers.get("mid") if isinstance(message_headers, Mapping) else None
                if incoming_mid and message.get("code") != 200:
                    ack = {
                        "code": 200,
                        "headers": {
                            "mid": incoming_mid,
                            "sid": message_headers.get("sid", ""),
                        },
                    }
                    for key in ("app-key", "ua", "dt"):
                        if key in message_headers:
                            ack["headers"][key] = message_headers[key]
                    await websocket.send(json.dumps(ack))

                if message.get("code") == 200 and str(incoming_mid or "") in pending_send_mids:
                    pending_send_mids.remove(str(incoming_mid))
                    report["send_acks"] += 1
                    _log("fixed_reply_ack", "received", request_hash=_digest(incoming_mid))

                body = message.get("body")
                package = body.get("syncPushPackage") if isinstance(body, Mapping) else None
                records = package.get("data") if isinstance(package, Mapping) else None
                if not isinstance(records, list):
                    continue

                for record in records:
                    raw_data = record.get("data") if isinstance(record, Mapping) else None
                    event = _decode_payload(raw_data, decrypt)
                    if event is None:
                        _log("sync_event", "decode_failed")
                        continue
                    chat = _extract_chat_event(event)
                    if chat is None:
                        continue

                    is_seller = chat["sender_id"] == seller_id
                    if is_seller:
                        if report["reply_submissions"] and chat["text"] == FIXED_REPLY:
                            report["self_echo_count"] += 1
                            _log(
                                "fixed_reply_echo",
                                "received",
                                chat_hash=_digest(chat["chat_id"]),
                            )
                        continue

                    report["buyer_messages"] += 1
                    _log(
                        "buyer_message",
                        "received",
                        chat_hash=_digest(chat["chat_id"]),
                        buyer_hash=_digest(chat["sender_id"]),
                        text_hash=_digest(chat["text"]),
                        text_length=len(chat["text"]),
                        trigger_match=chat["text"].strip() == trigger_text,
                    )
                    if report["reply_submissions"] >= expected_count or chat["text"].strip() != trigger_text:
                        report["ignored_buyer_messages"] += 1
                        continue

                    send_mid, payload = _send_message_payload(
                        chat_id=chat["chat_id"],
                        receiver_id=chat["sender_id"],
                        seller_id=seller_id,
                        text=FIXED_REPLY,
                        generate_mid=generate_mid,
                        generate_uuid=generate_uuid,
                    )
                    await websocket.send(payload)
                    report["reply_submissions"] += 1
                    report["status"] = "fixed_reply_submitted"
                    report["chat_hash"] = _digest(chat["chat_id"])
                    report["buyer_hash"] = _digest(chat["sender_id"])
                    pending_send_mids.add(send_mid)
                    if report["reply_submissions"] >= expected_count:
                        send_deadline = loop.time() + receipt_timeout
                    _log(
                        "fixed_reply",
                        "submitted_once",
                        reply=FIXED_REPLY,
                        request_hash=_digest(send_mid),
                        chat_hash=report["chat_hash"],
                        buyer_hash=report["buyer_hash"],
                    )

                if report["reply_submissions"] >= expected_count and not pending_send_mids:
                    break
    except Exception as exc:
        _log("websocket", "failed", error_type=type(exc).__name__, error=str(exc))
        report["status"] = "websocket_error"
        report["error_type"] = type(exc).__name__
    finally:
        stop_heartbeat.set()
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

    if report["reply_submissions"] >= expected_count:
        _log(
            "acceptance",
            "awaiting_buyer_window_confirmation",
            send_acks=report["send_acks"],
            self_echo_count=report["self_echo_count"],
        )
    else:
        _log("acceptance", "failed_no_fixed_reply", status=report["status"])
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--trigger-text", default=DEFAULT_TRIGGER)
    parser.add_argument("--listen-timeout", type=float, default=300.0)
    parser.add_argument("--receipt-timeout", type=float, default=30.0)
    parser.add_argument("--expected-count", type=int, default=1)
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


def main() -> int:
    global _LOG_FILE
    args = _parse_args()
    if args.listen_timeout <= 0 or args.receipt_timeout <= 0 or args.expected_count <= 0:
        raise SystemExit("timeouts and expected-count must be greater than zero")
    project_root = args.project_root.resolve()
    reference_root = args.reference_root.resolve()
    if args.log_file is not None:
        _LOG_FILE = args.log_file.resolve()
        _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    original_cwd = Path.cwd()
    try:
        with tempfile.TemporaryDirectory(prefix="xianyu-acceptance-") as isolated_cwd:
            try:
                os.chdir(isolated_cwd)
                report = asyncio.run(
                    _run(
                        project_root=project_root,
                        reference_root=reference_root,
                        trigger_text=args.trigger_text,
                        listen_timeout=args.listen_timeout,
                        receipt_timeout=args.receipt_timeout,
                        expected_count=args.expected_count,
                    )
                )
            finally:
                os.chdir(original_cwd)
    except Exception as exc:
        _log("acceptance", "failed", error_type=type(exc).__name__, error=str(exc))
        return 1

    print(json.dumps({"final_report": report}, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if report.get("reply_submissions", 0) >= args.expected_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
