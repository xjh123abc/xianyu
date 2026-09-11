"""Transport-neutral sending primitives for the Xianyu protocol."""

from __future__ import annotations

import base64
import json
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from app.channels.xianyu.models import SendReceipt


class TextSender(Protocol):
    async def send_text(self, chat_id: str, buyer_id: str, text: str, request_id: str) -> SendReceipt:
        ...


def build_text_payload(
    *,
    chat_id: str,
    buyer_id: str,
    seller_id: str,
    text: str,
    request_id: str,
) -> dict[str, Any]:
    if not chat_id or not buyer_id or not seller_id or not text.strip():
        raise ValueError("chat, buyer, seller and text are required")
    content = {"contentType": 1, "text": {"text": text}}
    encoded = base64.b64encode(json.dumps(content, ensure_ascii=False).encode("utf-8")).decode("ascii")
    return {
        "lwp": "/r/MessageSend/sendByReceiverScope",
        "headers": {"mid": request_id},
        "body": [
            {
                "uuid": str(uuid.uuid4()),
                "cid": f"{chat_id}@goofish",
                "conversationType": 1,
                "content": {"contentType": 101, "custom": {"type": 1, "data": encoded}},
                "redPointPolicy": 0,
                "extension": {"extJson": "{}"},
                "ctx": {"appVersion": "1.0", "platform": "web"},
                "mtags": {},
                "msgReadStatusSetting": 1,
            },
            {"actualReceivers": [f"{buyer_id}@goofish", f"{seller_id}@goofish"]},
        ],
    }


class WebSocketTextSender:
    """Single sender出口; receipt confirmation is supplied by the worker."""

    def __init__(self, websocket: Any, seller_id: str, *, send_json: Callable[[Any], Awaitable[None]] | None = None) -> None:
        self.websocket = websocket
        self.seller_id = seller_id
        self._send_json = send_json or self._default_send

    async def _default_send(self, payload: Any) -> None:
        await self.websocket.send(json.dumps(payload, ensure_ascii=False))

    async def send_text(self, chat_id: str, buyer_id: str, text: str, request_id: str) -> SendReceipt:
        payload = build_text_payload(
            chat_id=chat_id,
            buyer_id=buyer_id,
            seller_id=self.seller_id,
            text=text,
            request_id=request_id,
        )
        await self._send_json(payload)
        return SendReceipt(request_id=request_id, local_submitted=True)
