"""Translate reference WebSocket sync packages into normalized events."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from app.channels.xianyu.models import InboundMessage
from app.services.chat_contracts import ChatMessage


def to_chat_message(
    message: InboundMessage,
    *,
    item_id: str | None,
) -> ChatMessage:
    """Adapt a trusted Xianyu inbound event to the chat-core contract.

    ``item_id`` is supplied only after the worker verifies listing ownership.
    A raw Xianyu listing identifier must never become a trusted item identifier
    in the core business flow.
    """

    return ChatMessage(
        platform="xianyu",
        account_id=message.account_id,
        chat_id=message.chat_id,
        buyer_id=message.buyer_id,
        item_id=item_id,
        text=message.text,
    )


def decode_event(raw_data: object, decrypt: Callable[[str], str]) -> Mapping[str, Any] | None:
    if not isinstance(raw_data, str):
        return None
    try:
        decoded = base64.b64decode(raw_data).decode("utf-8")
        event = json.loads(decoded)
        return event if isinstance(event, Mapping) else None
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        pass
    try:
        event = json.loads(decrypt(raw_data))
    except (ValueError, TypeError, json.JSONDecodeError):
        return None
    return event if isinstance(event, Mapping) else None


def _stable_message_id(record: object, event: Mapping[str, Any]) -> str:
    if isinstance(record, Mapping):
        for key in ("messageId", "message_id", "msgId", "id", "pts"):
            value = record.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    first = event.get("1")
    details = first.get("10") if isinstance(first, Mapping) else None
    fallback = {
        "chat": first.get("2") if isinstance(first, Mapping) else None,
        "created": first.get("5") if isinstance(first, Mapping) else None,
        "sender": details.get("senderUserId") if isinstance(details, Mapping) else None,
        "text": details.get("reminderContent") if isinstance(details, Mapping) else None,
    }
    digest = hashlib.sha256(json.dumps(fallback, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return f"derived:{digest}"


def iter_sync_events(
    message: Mapping[str, Any],
    *,
    account_id: str,
    seller_id: str,
    decrypt: Callable[[str], str],
) -> Iterable[InboundMessage]:
    body = message.get("body")
    package = body.get("syncPushPackage") if isinstance(body, Mapping) else None
    records = package.get("data") if isinstance(package, Mapping) else None
    if not isinstance(records, list):
        return
    for record in records:
        raw_data = record.get("data") if isinstance(record, Mapping) else None
        event = decode_event(raw_data, decrypt)
        if event is None:
            continue
        first = event.get("1")
        details = first.get("10") if isinstance(first, Mapping) else None
        if not isinstance(first, Mapping) or not isinstance(details, Mapping):
            yield InboundMessage(
                account_id=account_id,
                platform_message_id=_stable_message_id(record, event),
                chat_id="unknown",
                buyer_id="unknown",
                text="",
                message_type="system",
                sender_is_seller=False,
                is_system_event=True,
            )
            continue

        platform_chat_id = str(first.get("2") or "").split("@", 1)[0]
        chat_id = f"xianyu:{seller_id}:{platform_chat_id}" if platform_chat_id else "unknown"
        sender_id = str(details.get("senderUserId") or "")
        text = details.get("reminderContent")
        reminder_url = str(details.get("reminderUrl") or "")
        item_id = None
        if "itemId=" in reminder_url:
            item_id = reminder_url.split("itemId=", 1)[1].split("&", 1)[0] or None
        sender_is_seller = sender_id == seller_id
        if not platform_chat_id or not sender_id:
            yield InboundMessage(
                account_id=account_id,
                platform_message_id=_stable_message_id(record, event),
                chat_id=chat_id or "unknown",
                buyer_id=sender_id or "unknown",
                text=str(text or ""),
                platform_chat_id=platform_chat_id or None,
                message_type="unknown",
                platform_item_id=item_id,
                sender_is_seller=sender_is_seller,
                is_system_event=True,
            )
            continue
        message_type = "text" if isinstance(text, str) and text.strip() else "attachment"
        yield InboundMessage(
            account_id=account_id,
            platform_message_id=_stable_message_id(record, event),
            chat_id=chat_id,
            buyer_id=sender_id,
            text=text if isinstance(text, str) else "",
            platform_chat_id=platform_chat_id,
            message_type=message_type,
            platform_item_id=item_id,
            sender_is_seller=sender_is_seller,
            is_system_event=False,
            received_at=str(first.get("5")) if first.get("5") is not None else None,
        )
