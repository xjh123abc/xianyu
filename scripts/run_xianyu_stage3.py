"""Run the S3 Xianyu buyer-message -> /chat -> reply loop.

The account stays disabled unless ``--enable`` is given.  A configured
Enterprise WeChat group-bot webhook is required because every unanswerable
question must notify the seller and switch its session to HUMAN.
"""

from __future__ import annotations

import argparse
import asyncio
import builtins
import hashlib
import inspect
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.channels.xianyu.adapter import iter_sync_events
from app.channels.xianyu.chat_client import ChatApiClient
from app.channels.xianyu.client import WebSocketTextSender
from app.channels.xianyu.stage3_worker import XianyuStage3Worker
from app.channels.xianyu.store import ChannelStore
from app.channels.xianyu.wecom import WeComWebhookNotifier
from probe_xianyu_channel import (
    DEFAULT_REFERENCE_COMMIT,
    DEFAULT_WS_URL,
    _install_reference_imports,
    _load_cookie,
    _registration_messages,
    _set_request_timeout,
)


def _digest(value: object) -> str | None:
    text = str(value or "").strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12] if text else None


def _log(path: Path, stage: str, result: str, **details: object) -> None:
    record = {
        "time": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "stage": stage,
        "result": result,
        **details,
    }
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


async def _heartbeat(websocket: Any, generate_mid: Any, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=15)
        except asyncio.TimeoutError:
            await websocket.send(json.dumps({"lwp": "/!", "headers": {"mid": str(generate_mid())}}))


async def run(args: argparse.Namespace) -> int:
    project_root = args.project_root.resolve()
    reference_root = args.reference_root.resolve()
    if not (reference_root / ".git").exists():
        raise ValueError("reference root must be the pinned local Xianyu template checkout")
    actual_commit = subprocess.run(
        ["git", "-C", str(reference_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if actual_commit != DEFAULT_REFERENCE_COMMIT:
        raise ValueError("reference checkout does not match the accepted pinned template commit")
    cookie_header, parsed_cookie, metadata = _load_cookie(project_root)
    values = dotenv_values(project_root / ".env")
    webhook = str(os.getenv("WECOM_WEBHOOK_URL") or values.get("WECOM_WEBHOOK_URL") or "").strip()
    if not webhook:
        raise ValueError("WECOM_WEBHOOK_URL is required before starting S3")
    _log(args.log_file, "cookie", "validated", configured=metadata["configured"], parsed_names=metadata["parsed_names"])

    _install_reference_imports(reference_root)
    from XianyuApis import XianyuApis
    from utils.xianyu_utils import decrypt, generate_device_id, generate_mid, generate_uuid

    api = XianyuApis()
    _set_request_timeout(api.session, timeout=15.0)
    api.session.cookies.update(dict(parsed_cookie))
    original_input = builtins.input
    builtins.input = lambda _prompt="": ""
    try:
        if not api.hasLogin():
            _log(args.log_file, "login", "failed")
            return 1
        seller_id = str(api.session.cookies.get("unb") or "")
        if not seller_id:
            raise ValueError("cookie has no seller account ID")
        # The token and WebSocket registration must use the exact same device
        # identity.  generate_device_id() is random, so it must be called once.
        device_id = generate_device_id(seller_id)
        token_result = api.get_token(device_id)
    finally:
        builtins.input = original_input
    token = token_result.get("data", {}).get("accessToken") if isinstance(token_result, Mapping) else None
    if not isinstance(token, str) or not token.strip():
        _log(args.log_file, "token", "failed")
        return 1
    account_id = args.account or seller_id
    store = ChannelStore(args.db)
    worker = XianyuStage3Worker(
        store,
        account_id=account_id,
        chat_client=ChatApiClient(args.chat_api, args.chat_timeout),
        notifier=WeComWebhookNotifier(webhook),
    )
    if args.enable:
        worker.enable_account()
    if not store.account_state(account_id)["enabled"]:
        raise RuntimeError("account is paused; run with --enable or use the control script to resume")

    import websockets

    refreshed_cookie = "; ".join(f"{name}={value}" for name, value in api.session.cookies.get_dict().items()) or cookie_header
    headers = {"Cookie": refreshed_cookie, "Host": "wss-goofish.dingtalk.com", "Origin": "https://www.goofish.com", "User-Agent": "Mozilla/5.0"}
    key = "additional_headers" if "additional_headers" in inspect.signature(websockets.connect).parameters else "extra_headers"
    registration, acknowledgement = _registration_messages(token, device_id, generate_mid)
    stop = asyncio.Event()
    try:
        async with websockets.connect(DEFAULT_WS_URL, open_timeout=15.0, **{key: headers}) as websocket:
            # The platform requires its reference client's message UUID format;
            # a standard RFC UUID can be written locally but rejected remotely.
            sender = WebSocketTextSender(
                websocket,
                seller_id,
                uuid_factory=generate_uuid,
                mid_factory=generate_mid,
            )
            await websocket.send(registration)
            await asyncio.sleep(1)
            await websocket.send(acknowledgement)
            _log(args.log_file, "websocket", "registered", account_hash=_digest(account_id))
            heartbeat = asyncio.create_task(_heartbeat(websocket, generate_mid, stop))
            try:
                while True:
                    raw = await websocket.recv()
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", errors="replace")
                    try:
                        payload = json.loads(raw)
                    except (TypeError, json.JSONDecodeError):
                        continue
                    if not isinstance(payload, Mapping):
                        continue
                    headers_in = payload.get("headers")
                    incoming_mid = headers_in.get("mid") if isinstance(headers_in, Mapping) else None
                    if incoming_mid and payload.get("code") != 200:
                        ack_headers = {"mid": incoming_mid, "sid": headers_in.get("sid", "")}
                        for header in ("app-key", "ua", "dt"):
                            if header in headers_in:
                                ack_headers[header] = headers_in[header]
                        await websocket.send(json.dumps({"code": 200, "headers": ack_headers}))
                    for event in iter_sync_events(payload, account_id=account_id, seller_id=seller_id, decrypt=decrypt):
                        result = await worker.process(event, sender)
                        _log(
                            args.log_file,
                            "buyer_event",
                            str(result.get("action", "unknown")),
                            message_hash=_digest(event.platform_message_id),
                            chat_hash=_digest(event.chat_id),
                            delivery=result.get("delivery"),
                            reason_hash=_digest(result.get("reason")),
                        )
            finally:
                stop.set()
                heartbeat.cancel()
                try:
                    await heartbeat
                except asyncio.CancelledError:
                    pass
    except Exception as exc:
        _log(args.log_file, "websocket", "failed", error_type=type(exc).__name__)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=Path("logs/xianyu_stage3.sqlite3"))
    parser.add_argument("--log-file", type=Path, default=Path("logs/xianyu_stage3.log"))
    parser.add_argument("--account", default="")
    parser.add_argument("--chat-api", default="http://127.0.0.1:8000")
    parser.add_argument("--chat-timeout", type=float, default=30.0)
    parser.add_argument("--enable", action="store_true", help="explicitly enable this account before listening")
    args = parser.parse_args()
    args.project_root = args.project_root.resolve()
    args.reference_root = args.reference_root.resolve()
    args.db = args.db.resolve()
    args.log_file = args.log_file.resolve()
    args.db.parent.mkdir(parents=True, exist_ok=True)
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    original_cwd = Path.cwd()
    try:
        # The reference client refreshes cookies relative to its working
        # directory.  Running from project_root lets it find the configured
        # .env instead of emitting a misleading “.env not found” warning.
        os.chdir(args.project_root)
        return asyncio.run(run(args))
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    raise SystemExit(main())
