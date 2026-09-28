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
from app.channels.xianyu.models import InboundMessage
from app.channels.xianyu.stage3_worker import XianyuStage3Worker
from app.channels.xianyu.item_sync import sync_bound_items
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
from app.channels.xianyu.runtime_supervisor import (
    BoundedEventProcessor,
    HeartbeatTracker,
    ReconnectPolicy,
    SerializedWebSocket,
    heartbeat_loop,
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


def _dev_test_chats(args: argparse.Namespace) -> set[str]:
    raw = getattr(args, "test_chat", None)
    values = raw if isinstance(raw, (list, tuple, set)) else [raw]
    return {str(value).strip() for value in values if str(value or "").strip()}


def _out_of_scope_reason(
    args: argparse.Namespace, event: Any, store: ChannelStore
) -> str | None:
    """Return an ignore reason before any out-of-scope event can call /chat."""

    if getattr(args, "dev_live", False):
        return None if event.chat_id in _dev_test_chats(args) else "dev_live_scope"
    if args.controlled_acceptance:
        if event.chat_id != args.acceptance_chat:
            return "controlled_acceptance_scope"
        if event.text.strip() not in args.acceptance_query and not (
            event.sender_is_seller
            and store.is_delivery_echo(args.account, event.chat_id, event.text)
        ):
            return "controlled_acceptance_scope"
    return None


async def _process_scoped_event(
    args: argparse.Namespace,
    event: Any,
    store: ChannelStore,
    worker: XianyuStage3Worker,
    sender: WebSocketTextSender,
    *,
    already_recorded: bool = False,
) -> Mapping[str, Any] | None:
    """Ignore other chats durably, or delegate one in-scope event unchanged."""

    reason = _out_of_scope_reason(args, event, store)
    if reason is not None:
        if not already_recorded and store.record_inbound(event):
            store.mark_ignored(event.account_id, event.platform_message_id, reason)
        return None
    return await worker.process(event, sender, already_recorded=already_recorded)


def _inbound_from_record(record: Mapping[str, Any]) -> InboundMessage:
    return InboundMessage(
        account_id=str(record["account_id"]),
        platform_message_id=str(record["platform_message_id"]),
        chat_id=str(record["chat_id"]),
        platform_chat_id=str(record["platform_chat_id"] or "") or None,
        buyer_id=str(record["buyer_id"]),
        text=str(record["text"] or ""),
        message_type=str(record["message_type"]),
        platform_item_id=str(record["platform_item_id"] or "") or None,
        sender_is_seller=bool(record["sender_is_seller"]),
        is_system_event=bool(record["is_system_event"]),
        received_at=str(record["received_at"] or "") or None,
    )


def _temporary_limit_reached(args: argparse.Namespace, processed: int) -> bool:
    if getattr(args, "dev_live", False):
        return args.dev_max_messages > 0 and processed >= args.dev_max_messages
    if args.controlled_acceptance:
        return processed >= args.acceptance_max_messages
    return False


async def _periodic_item_sync(
    api: Any, store: ChannelStore, *, account_id: str, snapshot_path: Path,
    interval_seconds: int, log_path: Path, stop: asyncio.Event,
) -> None:
    """Refresh already-bound listing snapshots while the channel stays online."""
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
            return
        except asyncio.TimeoutError:
            pass
        try:
            result = await asyncio.to_thread(
                sync_bound_items, api, store,
                account_id=account_id, snapshot_path=snapshot_path,
            )
            _log(log_path, "item_sync", "completed", synced=result["synced"], failed=len(result["failed"]))
        except Exception as error:
            _log(log_path, "item_sync", "failed", error_type=type(error).__name__)


async def _watch_stop_file(
    stop_path: Path, process_stop: asyncio.Event, receive_done: asyncio.Event
) -> None:
    while not process_stop.is_set():
        if stop_path.exists():
            process_stop.set()
            receive_done.set()
            return
        await asyncio.sleep(0.5)


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


async def run(args: argparse.Namespace) -> int:
    project_root = args.project_root.resolve()
    stop_path = project_root / "logs" / "xianyu_stage3.stop"
    stop_path.parent.mkdir(parents=True, exist_ok=True)
    stop_path.unlink(missing_ok=True)
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
    seller_id = str(api.session.cookies.get("unb") or "")
    if not seller_id:
        raise ValueError("cookie has no seller account ID")
    account_id = args.account or seller_id
    store = ChannelStore(args.db)
    recovery = store.recover_interrupted(account_id)
    _log(args.log_file, "process_recovery", "completed", **recovery)
    notifier = WeComWebhookNotifier(webhook)
    # The token and WebSocket registration must use the exact same device
    # identity.  generate_device_id() is random, so call it only once.
    device_id = generate_device_id(seller_id)
    auth_state, token = await asyncio.to_thread(
        _refresh_token_after_disconnect, api, device_id
    )
    if auth_state != "valid" or token is None:
        store.set_enabled(account_id, False)
        store.set_login_state(account_id, "LOGIN_REQUIRED")
        _log(args.log_file, "login", "reauthentication_required", state=auth_state)
        await _notify_runtime_failure(notifier, "xianyu_login_reauthentication_required", log_path=args.log_file)
        return 1
    store.set_login_state(account_id, "VALID")
    item_snapshot_path = resolve_project_path(
        settings.xianyu_items_path, project_root=project_root
    ).parent / "platform_items.json"
    try:
        sync_result = await asyncio.to_thread(
            sync_bound_items, api, store,
            account_id=account_id, snapshot_path=item_snapshot_path,
        )
        _log(
            args.log_file, "item_sync", "completed",
            bound=sync_result["bound"], synced=sync_result["synced"],
            failed=len(sync_result["failed"]),
        )
    except Exception as error:
        # Keep previously saved public data and continue serving from that snapshot.
        _log(args.log_file, "item_sync", "failed", error_type=type(error).__name__)
    worker = XianyuStage3Worker(
        store,
        account_id=account_id,
        chat_client=ChatApiClient(
            args.chat_api,
            args.chat_timeout,
            settings.xianyu_delivery_token,
        ),
        diagnostic_sink=lambda details: _log(
            args.log_file,
            "conversation_guard",
            str(details.get("decision", "unknown")).lower(),
            **dict(details),
        ),
    )
    temporary_mode = _temporary_mode(args)
    processed_messages = 0
    submitted_messages = 0
    process_stop = asyncio.Event()
    confirmation_deadline: float | None = None
    pending_confirmation_ids: list[str] = []
    try:
        with _account_activation(args, worker):
            if not store.account_state(account_id)["enabled"]:
                _log(args.log_file, "account_control", "paused_listen_only")

            import websockets

            reconnect_policy = ReconnectPolicy()
            while True:
                refreshed_cookie = "; ".join(f"{name}={value}" for name, value in api.session.cookies.get_dict().items()) or cookie_header
                headers = {"Cookie": refreshed_cookie, "Host": "wss-goofish.dingtalk.com", "Origin": "https://www.goofish.com", "User-Agent": "Mozilla/5.0"}
                key = "additional_headers" if "additional_headers" in inspect.signature(websockets.connect).parameters else "extra_headers"
                registration, acknowledgement = _registration_messages(token, device_id, generate_mid)
                stop = asyncio.Event()
                connection_started = asyncio.get_running_loop().time()
                try:
                    async with websockets.connect(DEFAULT_WS_URL, open_timeout=15.0, **{key: headers}) as websocket:
                        # The platform requires its reference client's message UUID format;
                        # a standard RFC UUID can be written locally but rejected remotely.
                        transport = SerializedWebSocket(websocket)
                        sender = WebSocketTextSender(
                            websocket,
                            seller_id,
                            send_json=lambda payload: transport.send(
                                json.dumps(payload, ensure_ascii=False)
                            ),
                            uuid_factory=generate_uuid,
                            mid_factory=generate_mid,
                        )
                        await websocket.send(registration)
                        await asyncio.sleep(1)
                        await websocket.send(acknowledgement)
                        _log(args.log_file, "websocket", "registered", account_hash=_digest(account_id))
                        heartbeat_tracker = HeartbeatTracker()
                        heartbeat = asyncio.create_task(
                            heartbeat_loop(
                                lambda payload: transport.send(
                                    json.dumps(payload, ensure_ascii=False)
                                ),
                                generate_mid,
                                heartbeat_tracker,
                                stop,
                            )
                        )
                        item_sync = asyncio.create_task(
                            _periodic_item_sync(
                                api,
                                store,
                                account_id=account_id,
                                snapshot_path=item_snapshot_path,
                                interval_seconds=max(60, int(settings.xianyu_item_sync_interval_seconds)),
                                log_path=args.log_file,
                                stop=stop,
                            )
                        )
                        receive_done = asyncio.Event()
                        stop_file_watcher = asyncio.create_task(
                            _watch_stop_file(stop_path, process_stop, receive_done)
                        )

                        async def handle_event(queued: tuple[InboundMessage, bool]) -> None:
                            nonlocal confirmation_deadline, processed_messages, submitted_messages
                            event, already_recorded = queued
                            result = await _process_scoped_event(
                                args,
                                event,
                                store,
                                worker,
                                sender,
                                already_recorded=already_recorded,
                            )
                            if result is None:
                                return
                            result_action = str(result.get("action", "unknown"))
                            _log(
                                args.log_file,
                                "delivery_confirmation"
                                if result_action == "delivery_confirmation"
                                else "buyer_event",
                                result_action,
                                message_hash=_digest(event.platform_message_id),
                                chat_hash=_digest(event.chat_id),
                                delivery=result.get("delivery"),
                                reason=result.get("reason"),
                                reason_hash=_digest(result.get("reason")),
                            )
                            if not temporary_mode or not _counts_acceptance_delivery(result):
                                return
                            delivery_state = result.get("delivery")
                            if delivery_state == "local_submitted":
                                submitted_messages += 1
                                pending_confirmation_ids.append(event.platform_message_id)
                                if _temporary_limit_reached(args, submitted_messages):
                                    confirmation_deadline = (
                                        asyncio.get_running_loop().time()
                                        + args.delivery_confirmation_timeout
                                    )
                                    _log(
                                        args.log_file,
                                        "delivery_confirmation",
                                        "waiting",
                                        timeout_seconds=args.delivery_confirmation_timeout,
                                    )
                            elif (
                                result_action == "delivery_confirmation"
                                or delivery_state == "confirmed"
                            ):
                                processed_messages += 1
                                confirmed_id = result.get("message_id")
                                if (
                                    isinstance(confirmed_id, str)
                                    and confirmed_id in pending_confirmation_ids
                                ):
                                    pending_confirmation_ids.remove(confirmed_id)
                                confirmation_deadline = (
                                    asyncio.get_running_loop().time()
                                    + args.delivery_confirmation_timeout
                                    if pending_confirmation_ids
                                    and _temporary_limit_reached(args, submitted_messages)
                                    else None
                                )
                            if _temporary_limit_reached(args, processed_messages):
                                receive_done.set()

                        def log_event_error(
                            queued: tuple[InboundMessage, bool], error: Exception
                        ) -> None:
                            event, _ = queued
                            _log(
                                args.log_file,
                                "buyer_event",
                                "processing_failed",
                                message_hash=_digest(event.platform_message_id),
                                error_type=type(error).__name__,
                            )

                        event_processor = BoundedEventProcessor(
                            handle_event,
                            capacity=128,
                            consumers=1,
                            on_error=log_event_error,
                        )
                        event_processor.start()
                        try:
                            await worker.recover_after_restart(sender)
                            for record in store.pending_inbound(account_id):
                                await event_processor.submit(
                                    (_inbound_from_record(record), True)
                                )
                            while True:
                                if confirmation_deadline is None:
                                    receive_task = asyncio.create_task(websocket.recv())
                                    stop_task = asyncio.create_task(receive_done.wait())
                                    completed, _ = await asyncio.wait(
                                        {receive_task, stop_task, heartbeat},
                                        return_when=asyncio.FIRST_COMPLETED,
                                    )
                                else:
                                    remaining = confirmation_deadline - asyncio.get_running_loop().time()
                                    if remaining <= 0:
                                        for pending_id in pending_confirmation_ids:
                                            await worker.mark_delivery_unconfirmed(pending_id)
                                        _log(
                                            args.log_file,
                                            "delivery_confirmation",
                                            "timeout",
                                            message="The platform did not echo the submitted reply.",
                                        )
                                        return 1
                                    receive_task = asyncio.create_task(websocket.recv())
                                    stop_task = asyncio.create_task(receive_done.wait())
                                    completed, _ = await asyncio.wait(
                                        {receive_task, stop_task, heartbeat},
                                        timeout=remaining,
                                        return_when=asyncio.FIRST_COMPLETED,
                                    )
                                    if not completed:
                                        receive_task.cancel()
                                        stop_task.cancel()
                                        for pending_id in pending_confirmation_ids:
                                            await worker.mark_delivery_unconfirmed(pending_id)
                                        _log(
                                            args.log_file,
                                            "delivery_confirmation",
                                            "timeout",
                                            message="The platform did not echo the submitted reply.",
                                        )
                                        return 1
                                if heartbeat in completed:
                                    heartbeat.result()
                                if stop_task in completed and receive_done.is_set():
                                    receive_task.cancel()
                                    await asyncio.gather(receive_task, return_exceptions=True)
                                    break
                                stop_task.cancel()
                                await asyncio.gather(stop_task, return_exceptions=True)
                                raw = receive_task.result()
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
                                if payload.get("code") == 200 and incoming_mid:
                                    heartbeat_tracker.acknowledge(incoming_mid)
                                queued_events: list[InboundMessage] = []
                                for event in iter_sync_events(
                                    payload,
                                    account_id=account_id,
                                    seller_id=seller_id,
                                    decrypt=decrypt,
                                ):
                                    out_of_scope = _out_of_scope_reason(args, event, store)
                                    inserted = store.record_inbound(event)
                                    if out_of_scope is not None:
                                        if inserted:
                                            store.mark_ignored(
                                                account_id,
                                                event.platform_message_id,
                                                out_of_scope,
                                            )
                                        continue
                                    existing = (
                                        None
                                        if inserted
                                        else store.message(
                                            account_id, event.platform_message_id
                                        )
                                    )
                                    if inserted or (
                                        existing is not None
                                        and existing.get("status")
                                        in {"RECEIVED", "CONTEXT_PENDING"}
                                    ):
                                        queued_events.append(event)
                                if incoming_mid and payload.get("code") != 200:
                                    ack_headers = {"mid": incoming_mid, "sid": headers_in.get("sid", "")}
                                    for header in ("app-key", "ua", "dt"):
                                        if header in headers_in:
                                            ack_headers[header] = headers_in[header]
                                    await transport.send(
                                        json.dumps({"code": 200, "headers": ack_headers})
                                    )
                                for event in queued_events:
                                    await event_processor.submit((event, True))
                        finally:
                            await event_processor.close()
                            stop_file_watcher.cancel()
                            await asyncio.gather(
                                stop_file_watcher, return_exceptions=True
                            )
                            stop.set()
                            heartbeat.cancel()
                            try:
                                await heartbeat
                            except asyncio.CancelledError:
                                pass
                            try:
                                await item_sync
                            except asyncio.CancelledError:
                                pass
                except Exception as exc:
                    _log(args.log_file, "websocket", "failed", error_type=type(exc).__name__)
                    if temporary_mode is not None:
                        return 1
                    failure_reason = type(exc).__name__
                else:
                    if temporary_mode is not None:
                        return 0
                    if process_stop.is_set():
                        _log(args.log_file, "runtime", "stopped")
                        return 0
                    failure_reason = "websocket_closed"
                reconnect_policy.note_disconnect(
                    asyncio.get_running_loop().time() - connection_started
                )
                if reconnect_policy.exhausted:
                    store.set_enabled(account_id, False)
                    store.set_login_state(account_id, "CONNECTION_FAILED")
                    _log(args.log_file, "reconnect", "exhausted", attempts=reconnect_policy.attempts, reason=failure_reason)
                    await _notify_runtime_failure(
                        notifier,
                        "xianyu_reconnect_exhausted",
                        log_path=args.log_file,
                    )
                    return 1
                auth_state, refreshed_token = await asyncio.to_thread(
                    _refresh_token_after_disconnect, api, device_id
                )
                if auth_state != "valid" or refreshed_token is None:
                    store.set_enabled(account_id, False)
                    store.set_login_state(account_id, "LOGIN_REQUIRED")
                    _log(args.log_file, "login", "reauthentication_required", state=auth_state)
                    await _notify_runtime_failure(
                        notifier,
                        "xianyu_login_reauthentication_required",
                        log_path=args.log_file,
                    )
                    return 1
                token = refreshed_token
                store.set_login_state(account_id, "VALID")
                delay_seconds = reconnect_policy.next_delay()
                _log(args.log_file, "reconnect", "scheduled", attempt=reconnect_policy.attempts, delay_seconds=delay_seconds)
                try:
                    await asyncio.wait_for(
                        process_stop.wait(), timeout=delay_seconds
                    )
                    _log(args.log_file, "runtime", "stopped")
                    return 0
                except asyncio.TimeoutError:
                    pass
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

    if getattr(args, "delivery_confirmation_timeout", 15.0) <= 0:
        raise ValueError("--delivery-confirmation-timeout must be greater than zero")
    if getattr(args, "dev_live", False):
        if args.enable:
            raise ValueError("--enable cannot be combined with --dev-live")
        if args.controlled_acceptance:
            raise ValueError("--controlled-acceptance cannot be combined with --dev-live")
        if not str(args.account or "").strip():
            raise ValueError("--account is required for dev-live")
        if not _dev_test_chats(args):
            raise ValueError("at least one --test-chat is required for dev-live")
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
    """Select successful sends and their later platform echo for acceptance."""

    action = result.get("action")
    if action == "delivery_confirmation":
        return result.get("delivery") == "confirmed"
    return action in {"answer", "human_handoff"} and result.get("delivery") in {
        "confirmed",
        "local_submitted",
    }


def _refresh_token_after_disconnect(api: Any, device_id: str) -> tuple[str, str | None]:
    """Refresh channel authorization without allowing the reference client to prompt."""

    original_input = builtins.input
    builtins.input = lambda _prompt="": ""
    try:
        if not api.hasLogin():
            return "login_invalid", None
        try:
            result = api.get_token(device_id)
        except SystemExit:
            return "token_unavailable", None
        token = result.get("data", {}).get("accessToken") if isinstance(result, Mapping) else None
        if not isinstance(token, str) or not token.strip():
            return "token_unavailable", None
        return "valid", token
    except Exception:
        return "token_unavailable", None
    finally:
        builtins.input = original_input


async def _notify_runtime_failure(notifier: Any, reason: str, *, log_path: Path) -> None:
    try:
        await notifier.notify_handoff(
            chat_id="channel-runtime",
            item_id=None,
            reason=reason,
            question="闲鱼客服运行状态需要人工检查。",
        )
    except Exception as error:
        _log(
            log_path,
            "runtime_alert",
            "failed",
            reason=reason,
            error_type=type(error).__name__,
        )
    else:
        _log(log_path, "runtime_alert", "sent", reason=reason)


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
        "--delivery-confirmation-timeout",
        type=float,
        default=15.0,
        help="seconds to keep the Xianyu socket open for a seller-side send echo",
    )
    parser.add_argument(
        "--dev-live",
        action="store_true",
        help="temporarily process arbitrary buyer text from explicitly named test chats",
    )
    parser.add_argument(
        "--test-chat",
        action="append",
        default=[],
        help="test conversation to allow in dev-live; repeat for multiple chats",
    )
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
