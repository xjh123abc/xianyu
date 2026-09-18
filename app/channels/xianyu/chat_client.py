"""A narrow HTTP client for the already-existing local /chat API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import httpx

from app.services.chat_contracts import ChatMessage


class ChatClient(Protocol):
    async def ask(self, message: ChatMessage) -> Mapping[str, Any]:
        ...


class ChatApiClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout_seconds: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def ask(self, message: ChatMessage) -> Mapping[str, Any]:
        """Send one unified message through the unchanged /chat wire format."""

        body: dict[str, Any] = {"query": message.text, "chat_id": message.chat_id}
        if message.item_id:
            body["item_id"] = message.item_id
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(f"{self.base_url}/chat", json=body)
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError("/chat response must be a JSON object")
        return payload
