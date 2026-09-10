"""In-memory short-term session history and business state."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import uuid4


class SessionManager:
    """Store recent chat turns and small structured state per ``chat_id``."""

    def __init__(self, max_history: int = 10) -> None:
        if max_history <= 0:
            raise ValueError("max_history must be greater than zero")
        self.max_history = max_history
        self.sessions: dict[str, dict[str, Any]] = {}

    def get_or_create(self, chat_id: str | None = None) -> tuple[str, dict[str, Any]]:
        """Return an isolated session, creating it when necessary."""
        normalized_id = chat_id.strip() if isinstance(chat_id, str) else ""
        if not normalized_id:
            normalized_id = str(uuid4())

        session = self.sessions.setdefault(
            normalized_id,
            {
                "history": [],
                "state": {
                    "order_id": None,
                    "item_id": None,
                    "current_item_id": None,
                    "last_intent": None,
                },
            },
        )
        return normalized_id, session

    def set_current_item_id(self, chat_id: str, item_id: str) -> None:
        """Set the confirmed current item for exactly one chat session."""
        normalized_item_id = item_id.strip().upper()
        if not normalized_item_id:
            raise ValueError("item_id must not be empty")

        _, session = self.get_or_create(chat_id)
        session["state"]["current_item_id"] = normalized_item_id
        # Keep the stage-3 state alias until its separate migration stage.
        session["state"]["item_id"] = normalized_item_id

    def get_current_item_id(self, chat_id: str) -> str | None:
        """Return the current item stored for one chat without crossing sessions."""
        _, session = self.get_or_create(chat_id)
        current_item_id = session["state"].get("current_item_id")
        if not current_item_id:
            current_item_id = session["state"].get("item_id")
        return str(current_item_id) if current_item_id else None

    def append_turn(
        self,
        chat_id: str,
        user_message: str,
        assistant_message: str,
        *,
        order_id: str | None = None,
        item_id: str | None = None,
        last_intent: str | None = None,
    ) -> None:
        """Save one user/assistant turn and update known business state."""
        _, session = self.get_or_create(chat_id)
        history = session["history"]
        history.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ]
        )
        del history[:-self.max_history]

        state = session["state"]
        if order_id:
            state["order_id"] = order_id
        if item_id:
            self.set_current_item_id(chat_id, item_id)
        if last_intent:
            state["last_intent"] = last_intent

    @staticmethod
    def read_context(session: Mapping[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """Return defensive copies of history and state for one request."""
        history = [dict(item) for item in session.get("history", [])]
        state = dict(session.get("state", {}))
        return history, state
