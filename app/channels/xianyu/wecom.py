"""Enterprise WeChat group-bot notification adapter for human handoffs."""

from __future__ import annotations

from typing import Protocol
from urllib.parse import urlparse

import httpx


class HandoffNotifier(Protocol):
    async def notify_handoff(self, *, chat_id: str, item_id: str | None, reason: str, question: str) -> None:
        ...


class WeComWebhookNotifier:
    """Post a minimal human-handoff notice without logging the secret URL."""

    def __init__(self, webhook_url: str, *, timeout_seconds: float = 10.0) -> None:
        parsed = urlparse(webhook_url)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "qyapi.weixin.qq.com"
            or parsed.path != "/cgi-bin/webhook/send"
            or not parsed.query
        ):
            raise ValueError("WECOM_WEBHOOK_URL must be an https qyapi.weixin.qq.com webhook URL")
        self._webhook_url = webhook_url
        self._timeout_seconds = timeout_seconds

    @staticmethod
    def _content(*, chat_id: str, item_id: str | None, reason: str, question: str) -> str:
        safe_question = " ".join(question.split())[:500]
        safe_reason = " ".join(reason.split())[:200]
        item = item_id or "未识别商品"
        return f"闲鱼客服待人工处理\n会话：{chat_id}\n商品：{item}\n原因：{safe_reason}\n买家问题：{safe_question}"

    async def notify_handoff(self, *, chat_id: str, item_id: str | None, reason: str, question: str) -> None:
        payload = {"msgtype": "text", "text": {"content": self._content(chat_id=chat_id, item_id=item_id, reason=reason, question=question)}}
        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.post(self._webhook_url, json=payload)
            response.raise_for_status()
            data = response.json()
        if not isinstance(data, dict) or data.get("errcode") != 0:
            raise RuntimeError("Enterprise WeChat notification was rejected")


class DisabledNotifier:
    """Explicit local configuration failure; never claims the seller was notified."""

    async def notify_handoff(self, *, chat_id: str, item_id: str | None, reason: str, question: str) -> None:
        raise RuntimeError("Enterprise WeChat webhook is not configured")
