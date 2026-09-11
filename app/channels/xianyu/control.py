"""Small programmatic control surface for the S2 worker."""

from __future__ import annotations

from typing import Any

from app.channels.xianyu.store import ChannelStore


class ChannelControl:
    def __init__(self, store: ChannelStore, account_id: str) -> None:
        self.store = store
        self.account_id = account_id

    def status(self) -> dict[str, Any]:
        return self.store.account_state(self.account_id)

    def pause(self) -> int:
        return self.store.set_enabled(self.account_id, False)

    def resume(self) -> int:
        return self.store.set_enabled(self.account_id, True)

    def takeover(self, chat_id: str, buyer_id: str, reason: str = "seller_takeover") -> int:
        return self.store.set_session_mode(self.account_id, chat_id, buyer_id, "HUMAN", reason)

    def release(self, chat_id: str, buyer_id: str) -> int:
        return self.store.set_session_mode(self.account_id, chat_id, buyer_id, "AUTO")
