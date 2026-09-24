"""Run the S3 Xianyu buyer-message -> /chat -> reply loop.

The account stays disabled unless a production enable or an explicitly scoped
temporary mode is given. A configured Enterprise WeChat group-bot webhook is
required because every unanswerable question must notify the seller and switch
its session to HUMAN.
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
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.channels.xianyu.adapter import iter_sync_events
from app.channels.xianyu.acceptance_gate import load_s6_acceptance_report
from app.channels.xianyu.chat_client import ChatApiClient
from app.channels.xianyu.client import WebSocketTextSender
from app.channels.xianyu.stage3_worker import XianyuStage3Worker
from app.channels.xianyu.store import ChannelStore
from app.channels.xianyu.wecom import WeComWebhookNotifier
from config.paths import resolve_project_path
from config.settings import settings
from app.channels.xianyu.reference_runtime import (
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


def resolve_channel_database_path(
    database_path: Path | None,
    *,
    project_root: Path,
) -> Path:
    """Resolve an explicit override or the shared channel-database setting."""

    configured_path = (
        database_path
        if database_path is not None
        else settings.xianyu_channel_database_path
    )
    return resolve_project_path(configured_path, project_root=project_root)


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


def _temporary_mode(args: argparse.Namespace) -> str | None:
    if getattr(args, "dev_live", False):
        return "dev_live"
    if args.controlled_acceptance:
        return "controlled_acceptance"
    return None


def _out_of_scope_reason(args: argparse.Namespace, event: Any) -> str | None:
    """Return an ignore reason before any out-of-scope event can call /chat."""

    if getattr(args, "dev_live", False):
        return None if event.chat_id == args.test_chat else "dev_live_scope"
    if args.controlled_acceptance and (
        event.chat_id != args.acceptance_chat
        or event.text.strip() not in args.acceptance_query
    ):
        return "controlled_acceptance_scope"
    return None


async def _process_scoped_event(
    args: argparse.Namespace,
    event: Any,
    store: ChannelStore,
    worker: XianyuStage3Worker,
    sender: WebSocketTextSender,
) -> Mapping[str, Any] | None:
    """Ignore other chats durably, or delegate one in-scope event unchanged."""

    reason = _out_of_scope_reason(args, event)
    if reason is not None:
        if store.record_inbound(event):
            store.mark_ignored(args.account, event.platform_message_id, reason)
        return None
    return await worker.process(event, sender)


def _temporary_limit_reached(args: argparse.Namespace, processed: int) -> bool:
    if getattr(args, "dev_live", False):
        return args.dev_max_messages > 0 and processed >= args.dev_max_messages
    if args.controlled_acceptance:
        return processed >= args.acceptance_max_messages
    return False


def _current_project_commit(args: argparse.Namespace, project_root: Path) -> str | None:
    """Bind production/acceptance to HEAD while leaving dev-live unbound."""

    if getattr(args, "dev_live", False):
        return None
    return subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()


@contextmanager
def _account_activation(args: argparse.Namespace, worker: XianyuStage3Worker):
    """Enable temporary modes and always restore their account to paused."""

    temporary_mode = _temporary_mode(args)
    if args.enable or temporary_mode is not None:
        worker.enable_account()
    try:
        yield
    finally:
        if temporary_mode is not None:
            worker.pause_account()


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
    current_commit = _current_project_commit(args, project_root)
    _authorise_start(args, current_commit)
    cookie_header, parsed_cookie, metadata = _load_cookie(project_root)
    values = dotenv_values(project_root / ".env")
    webhook = str(os.getenv("WECOM_WEBHOOK_URL") or values.get("WECOM_WEBHOOK_URL") or "").strip()
    if not webhook:
        raise ValueError("WECOM_WEBHOOK_URL is required before starting S3")
    if not settings.xianyu_delivery_token:
        raise ValueError("XIANYU_DELIVERY_TOKEN is required before starting T1 delivery reporting")
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
        chat_client=ChatApiClient(
            args.chat_api,
            args.chat_timeout,
            settings.xianyu_delivery_token,
        ),
        notifier=WeComWebhookNotifier(webhook),
        diagnostic_sink=lambda details: _log(
            args.log_file,
            "conversation_guard",
            str(details.get("decision", "unknown")).lower(),
            **dict(details),
        ),
    )
    temporary_mode = _temporary_mode(args)
    processed_messages = 0
    try:
        with _account_activation(args, worker):
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
                                result = await _process_scoped_event(
                                    args, event, store, worker, sender
                                )
                                if result is None:
                                    continue
                                _log(
                                    args.log_file,
                                    "buyer_event",
                                    str(result.get("action", "unknown")),
                                    message_hash=_digest(event.platform_message_id),
                                    chat_hash=_digest(event.chat_id),
                                    delivery=result.get("delivery"),
                                    reason=result.get("reason"),
                                    reason_hash=_digest(result.get("reason")),
                                )
                                if temporary_mode and _counts_acceptance_delivery(result):
                                    processed_messages += 1
                                    if _temporary_limit_reached(args, processed_messages):
                                        return 0
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
        return 0
    finally:
        if temporary_mode is not None:
            _log(
                args.log_file,
                temporary_mode,
                "paused",
                processed=processed_messages,
            )


def _authorise_start(args: argparse.Namespace, current_commit: str | None) -> None:
    """Allow either a tightly scoped trial or a fully approved production run."""

    if getattr(args, "dev_live", False):
        if args.enable:
            raise ValueError("--enable cannot be combined with --dev-live")
        if args.controlled_acceptance:
            raise ValueError("--controlled-acceptance cannot be combined with --dev-live")
        if not str(args.account or "").strip():
            raise ValueError("--account is required for dev-live")
        if not str(args.test_chat or "").strip():
            raise ValueError("--test-chat is required for dev-live")
        if args.dev_max_messages < 0:
            raise ValueError("--dev-max-messages must be zero or greater")
        return
    if args.controlled_acceptance:
        if args.enable:
            raise ValueError("--enable cannot be combined with --controlled-acceptance")
        if not str(args.account or "").strip():
            raise ValueError("--account is required for controlled acceptance")
        if not str(args.acceptance_chat or "").strip():
            raise ValueError("--acceptance-chat is required for controlled acceptance")
        if not args.acceptance_query:
            raise ValueError("at least one --acceptance-query is required")
        if args.acceptance_max_messages <= 0:
            raise ValueError("--acceptance-max-messages must be greater than zero")
        return
    if args.acceptance_report is None:
        raise ValueError("--acceptance-report is required before real automatic sending")
    if current_commit is None:
        raise ValueError("current Git commit is required before real automatic sending")
    load_s6_acceptance_report(
        args.acceptance_report,
        expected_commit=current_commit,
    )


def _counts_acceptance_delivery(result: Mapping[str, object]) -> bool:
    """Count only a buyer-visible submission, never duplicates or failed sends."""

    return result.get("action") in {"answer", "human_handoff"} and result.get(
        "delivery"
    ) in {"confirmed", "local_submitted"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument(
        "--db",
        type=Path,
        help="explicit channel SQLite path; overrides XIANYU_CHANNEL_DATABASE_PATH",
    )
    parser.add_argument("--log-file", type=Path, default=Path("logs/xianyu_stage3.log"))
    parser.add_argument("--account", default="")
    parser.add_argument("--chat-api", default="http://127.0.0.1:8000")
    parser.add_argument("--chat-timeout", type=float, default=30.0)
    parser.add_argument(
        "--acceptance-report",
        type=Path,
        help="S6 report approving this exact Git commit for real automatic sending",
    )
    parser.add_argument(
        "--controlled-acceptance",
        action="store_true",
        help="temporarily enable only one explicitly named test conversation",
    )
    parser.add_argument("--acceptance-chat", default="")
    parser.add_argument(
        "--acceptance-query",
        action="append",
        default=[],
        help="exact buyer text allowed in controlled acceptance; repeat as needed",
    )
    parser.add_argument("--acceptance-max-messages", type=int, default=1)
    parser.add_argument(
        "--dev-live",
        action="store_true",
        help="temporarily process arbitrary buyer text from exactly one test chat",
    )
    parser.add_argument("--test-chat", default="")
    parser.add_argument(
        "--dev-max-messages",
        type=int,
        default=0,
        help="stop after this many delivered test-chat replies; zero runs until interrupted",
    )
    parser.add_argument("--enable", action="store_true", help="explicitly enable this account before listening")
    args = parser.parse_args()
    args.project_root = args.project_root.resolve()
    args.reference_root = args.reference_root.resolve()
    args.db_source = "--db" if args.db is not None else "XIANYU_CHANNEL_DATABASE_PATH"
    args.db = resolve_channel_database_path(args.db, project_root=args.project_root)
    args.log_file = args.log_file.resolve()
    args.db.parent.mkdir(parents=True, exist_ok=True)
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "database_path": str(args.db),
                "database_source": args.db_source,
                "stage": "database",
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
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
