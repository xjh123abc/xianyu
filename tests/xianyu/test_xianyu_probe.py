"""Unit tests for the receive-only Xianyu channel probe."""

from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path

import pytest

from app.channels.xianyu.reference_runtime import (
    _decode_payload,
    _load_cookie,
    _registration_messages,
    _set_request_timeout,
)
from scripts.probe_xianyu_channel import (
    _hash_identifier,
    _record_message,
)


def test_probe_hashes_identifiers_and_preserves_empty_values() -> None:
    assert _hash_identifier(None) is None
    assert _hash_identifier("   ") is None
    assert _hash_identifier("buyer-1") == _hash_identifier("buyer-1")
    assert _hash_identifier("buyer-1") != _hash_identifier("buyer-2")


def test_probe_decodes_plain_base64_json_without_raw_logging() -> None:
    event = {"operation": "sessionUpdate", "sessionId": "session-1"}
    encoded = base64.b64encode(json.dumps(event).encode("utf-8")).decode("ascii")

    assert _decode_payload(encoded, lambda value: "unused") == event


def test_shared_reference_helpers_preserve_registration_and_timeout_contract() -> None:
    calls: list[tuple[str, str, dict[str, object]]] = []

    class Session:
        @staticmethod
        def request(method: str, url: str, **kwargs: object) -> str:
            calls.append((method, url, kwargs))
            return "response"

    session = Session()
    _set_request_timeout(session, timeout=12.0)
    assert session.request("GET", "https://example.test") == "response"
    assert session.request("POST", "https://example.test", timeout=3.0) == "response"
    assert calls == [
        ("GET", "https://example.test", {"timeout": 12.0}),
        ("POST", "https://example.test", {"timeout": 3.0}),
    ]

    registration, acknowledgement = _registration_messages(
        "token-value", "device-value", lambda: "mid-value"
    )
    registration_payload = json.loads(registration)
    acknowledgement_payload = json.loads(acknowledgement)
    assert registration_payload["lwp"] == "/reg"
    assert registration_payload["headers"]["token"] == "token-value"
    assert registration_payload["headers"]["did"] == "device-value"
    assert registration_payload["headers"]["mid"] == "mid-value"
    assert acknowledgement_payload["lwp"] == "/r/SyncStatus/ackDiff"
    assert acknowledgement_payload["body"][0]["pipeline"] == "sync"


def test_probe_classifies_buyer_event_and_counts_sync_package() -> None:
    event = {
        "1": {
            "2": "chat-1@goofish",
            "5": 1700000000000,
            "10": {
                "senderUserId": "buyer-1",
                "reminderContent": "阶段一测试",
                "reminderUrl": "https://www.goofish.com/item?itemId=ITEM-1",
                "contentType": 1,
            },
        }
    }
    encoded = base64.b64encode(json.dumps(event).encode("utf-8")).decode("ascii")
    message = {"body": {"syncPushPackage": {"data": [{"data": encoded}]}}}
    report = {
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

    _record_message(message, "seller-1", lambda value: "unused", report)

    assert report["sync_packages"] == 1
    assert report["decoded_events"] == 1
    assert report["buyer_messages"] == 1
    assert report["seller_messages"] == 0
    assert report["events"][0]["kind"] == "buyer_chat"
    assert report["events"][0]["text_length"] == 5
    assert report["events"][0]["chat_id_hash"]
    assert report["events"][0]["item_id_hash"]
    assert "阶段一测试" not in json.dumps(report, ensure_ascii=False)


def test_probe_rejects_missing_or_malformed_cookie(monkeypatch: pytest.MonkeyPatch) -> None:
    test_root = Path(__file__).parent / ".xianyu-probe-cookie-test"
    if test_root.exists():
        shutil.rmtree(test_root)
    test_root.mkdir()
    try:
        (test_root / ".env").write_text(
            "XIANYU_COOKIES_STR=not-a-cookie\n",
            encoding="utf-8",
        )
        monkeypatch.delenv("XIANYU_COOKIES_STR", raising=False)

        with pytest.raises(ValueError, match="complete browser cookie"):
            _load_cookie(test_root)
    finally:
        if test_root.exists():
            shutil.rmtree(test_root)
