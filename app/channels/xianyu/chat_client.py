"""A narrow HTTP client for the already-existing local /chat API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import httpx

from app.services.chat_contracts import ChatMessage


class ChatClient(Protocol):
    async def ask(self, message: ChatMessage) -> Mapping[str, Any]:
        ...

    async def report_delivery(
        self,
        *,
        chat_id: str,
        turn_id: str,
        proposal_id: str,
        delivery_state: str,
    ) -> Mapping[str, Any]:
        ...

    async def record_conversation_event(
        self, *, account_id: str, chat_id: str, event_id: str, role: str,
        source: str, content: str
    ) -> Mapping[str, Any]:
        ...


class ChatApiClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        timeout_seconds: float = 30.0,
        delivery_token: str = "",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.delivery_token = delivery_token

    async def ask(self, message: ChatMessage) -> Mapping[str, Any]:
        """Send one unified message through the unchanged /chat wire format."""

        body: dict[str, Any] = {"query": message.text, "chat_id": message.chat_id}
        if message.item_id:
            body["item_id"] = message.item_id
        if message.turn_id:
            body["turn_id"] = message.turn_id
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(f"{self.base_url}/chat", json=body)
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError("/chat response must be a JSON object")
        return payload

    async def report_delivery(
        self,
        *,
        chat_id: str,
        turn_id: str,
        proposal_id: str,
        delivery_state: str,
    ) -> Mapping[str, Any]:
        """Report an existing channel result without asking it to resend text."""

        if not self.delivery_token:
            raise RuntimeError("XIANYU_DELIVERY_TOKEN is not configured")
        body = {
            "chat_id": chat_id,
            "turn_id": turn_id,
            "proposal_id": proposal_id,
            "delivery_state": delivery_state,
        }
        headers = {"X-Internal-Token": self.delivery_token}
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/internal/chat/delivery",
                json=body,
                headers=headers,
            )
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError("delivery response must be a JSON object")
        return payload

    async def record_conversation_event(
        self, *, account_id: str, chat_id: str, event_id: str, role: str,
        source: str, content: str
    ) -> Mapping[str, Any]:
        """Append a channel event to chat history without generating a reply."""
        if not self.delivery_token:
            raise RuntimeError("XIANYU_DELIVERY_TOKEN is not configured")
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/internal/chat/events",
                json={
                    "account_id": account_id,
                    "chat_id": chat_id,
                    "event_id": event_id,
                    "role": role,
                    "source": source,
                    "content": content,
                },
                headers={"X-Internal-Token": self.delivery_token},
            )
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError("conversation event response must be a JSON object")
        return payload
