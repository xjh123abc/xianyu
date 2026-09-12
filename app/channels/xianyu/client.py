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
    message_uuid: str | None = None,
    platform_mid: str | None = None,
) -> dict[str, Any]:
    if not chat_id or not buyer_id or not seller_id or not text.strip():
        raise ValueError("chat, buyer, seller and text are required")
    content = {"contentType": 1, "text": {"text": text}}
    encoded = base64.b64encode(json.dumps(content, ensure_ascii=False).encode("utf-8")).decode("ascii")
    return {
        "lwp": "/r/MessageSend/sendByReceiverScope",
        # request_id is our durable idempotency key; the Xianyu protocol has
        # its own timestamp-shaped message ID in the wire header.
        "headers": {"mid": platform_mid or request_id},
        "body": [
            {
                "uuid": message_uuid or str(uuid.uuid4()),
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

    def __init__(
        self,
        websocket: Any,
        seller_id: str,
        *,
        send_json: Callable[[Any], Awaitable[None]] | None = None,
        uuid_factory: Callable[[], str] | None = None,
        mid_factory: Callable[[], str] | None = None,
    ) -> None:
        self.websocket = websocket
        self.seller_id = seller_id
        self._send_json = send_json or self._default_send
        self._uuid_factory = uuid_factory or (lambda: str(uuid.uuid4()))
        self._mid_factory = mid_factory or (lambda: str(uuid.uuid4()))

    async def _default_send(self, payload: Any) -> None:
        await self.websocket.send(json.dumps(payload, ensure_ascii=False))

    async def send_text(self, chat_id: str, buyer_id: str, text: str, request_id: str) -> SendReceipt:
        payload = build_text_payload(
            chat_id=chat_id,
            buyer_id=buyer_id,
            seller_id=self.seller_id,
            text=text,
            request_id=request_id,
            message_uuid=self._uuid_factory(),
            platform_mid=self._mid_factory(),
        )
        await self._send_json(payload)
        return SendReceipt(request_id=request_id, local_submitted=True)
