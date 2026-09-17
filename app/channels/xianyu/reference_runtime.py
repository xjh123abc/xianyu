"""Pinned Xianyu reference-client helpers shared by channel entry points."""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from collections.abc import Mapping
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Any

from dotenv import dotenv_values


DEFAULT_REFERENCE_COMMIT = "540bbc26cf02ee6348d997843942776a9be9460b"
DEFAULT_WS_URL = "wss://wss-goofish.dingtalk.com/"


def _load_cookie(project_root: Path) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Load and validate the local cookie without returning it in reports."""

    values = dotenv_values(project_root / ".env")
    raw = str(values.get("XIANYU_COOKIES_STR") or os.getenv("XIANYU_COOKIES_STR") or "")
    cookie = SimpleCookie()
    cookie.load(raw)
    parsed = {name: morsel.value for name, morsel in cookie.items()}
    metadata = {
        "configured": bool(raw),
        "length": len(raw),
        "segments": len([part for part in raw.split(";") if part.strip()]),
        "parsed_names": len(parsed),
        "has_unb": "unb" in parsed,
        "has_cookie2": "cookie2" in parsed,
        "has_m_h5_tk": "_m_h5_tk" in parsed,
        "has_cna": "cna" in parsed,
    }
    required = ("unb", "cookie2", "_m_h5_tk")
    if not raw or any(name not in parsed for name in required):
        raise ValueError(
            "XIANYU_COOKIES_STR must be a complete browser cookie string "
            "containing unb, cookie2 and _m_h5_tk"
        )
    normalized = "; ".join(f"{name}={value}" for name, value in parsed.items())
    return normalized, parsed, metadata


def _install_reference_imports(reference_root: Path) -> None:
    """Make only the pinned reference transport modules importable."""

    if not (reference_root / "XianyuApis.py").is_file():
        raise FileNotFoundError(f"reference checkout is missing XianyuApis.py: {reference_root}")
    if not (reference_root / "utils" / "xianyu_utils.py").is_file():
        raise FileNotFoundError(f"reference checkout is missing xianyu_utils.py: {reference_root}")
    sys.path.insert(0, str(reference_root))


def _set_request_timeout(session: Any, timeout: float) -> None:
    """Add a bounded timeout to the reference API's otherwise-unbounded calls."""

    original_request = session.request

    def request(method: str, url: str, **kwargs: Any) -> Any:
        kwargs.setdefault("timeout", timeout)
        return original_request(method, url, **kwargs)

    session.request = request


def _registration_messages(token: str, device_id: str, generate_mid: Any) -> tuple[str, str]:
    """Build registration and sync acknowledgement messages."""

    registration = {
        "lwp": "/reg",
        "headers": {
            "cache-header": "app-key token ua wv",
            "app-key": "444e9908a51d1cb236a27862abc769c9",
            "token": token,
            "ua": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/133.0.0.0 Safari/537.36",
            "dt": "j",
            "wv": "im:3,au:3,sy:6",
            "sync": "0,0;0;0;",
            "did": device_id,
            "mid": generate_mid(),
        },
    }
    acknowledgement = {
        "lwp": "/r/SyncStatus/ackDiff",
        "headers": {"mid": "5701741704675979 0"},
        "body": [
            {
                "pipeline": "sync",
                "tooLong2Tag": "PNM,1",
                "channel": "sync",
                "topic": "sync",
                "highPts": 0,
                "pts": int(time.time() * 1000) * 1000,
                "seq": 0,
                "timestamp": int(time.time() * 1000),
            }
        ],
    }
    return json.dumps(registration), json.dumps(acknowledgement)


def _decode_payload(raw_data: object, decrypt: Any) -> Mapping[str, Any] | None:
    """Decode plaintext or reference MessagePack payloads, without logging raw data."""

    if not isinstance(raw_data, str):
        return None
    try:
        decoded = base64.b64decode(raw_data).decode("utf-8")
        payload = json.loads(decoded)
        return payload if isinstance(payload, Mapping) else None
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        pass
    try:
        payload = json.loads(decrypt(raw_data))
    except (ValueError, TypeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, Mapping) else None
