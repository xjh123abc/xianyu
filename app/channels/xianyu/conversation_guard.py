"""Fail-closed eligibility check for seller-side AI customer service."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from collections.abc import Callable, Mapping
from typing import Literal

from app.channels.xianyu.models import InboundMessage
from app.channels.xianyu.store import ChannelStore


logger = logging.getLogger(__name__)


GuardAction = Literal["ALLOW", "IGNORE"]
DiagnosticSink = Callable[[Mapping[str, object]], None]


@dataclass(frozen=True, slots=True)
class ConversationDecision:
    """Whether an inbound event belongs to this seller's AI service."""

    action: GuardAction
    reason: str


class ConversationGuard:
    """Allow only buyer messages about a listing bound to this seller account."""

    def __init__(
        self,
        store: ChannelStore,
        *,
        account_id: str,
        diagnostic_sink: DiagnosticSink | None = None,
    ) -> None:
        self.store = store
        self.account_id = account_id
        self.diagnostic_sink = diagnostic_sink

    @staticmethod
    def _hash_identifier(value: object) -> str | None:
        text = str(value or "").strip()
        if not text:
            return None
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

    def _finish(
        self,
        message: InboundMessage,
        *,
        action: GuardAction,
        reason: str,
        mapped_item_id: str | None,
        is_item_bound: bool,
        item_bound_checked: bool,
        item_source: str = "none",
    ) -> ConversationDecision:
        diagnostic: dict[str, object] = {
            "chat_hash": self._hash_identifier(message.chat_id),
            "message_hash": self._hash_identifier(message.platform_message_id),
            "message_type": message.message_type,
            "is_system_event": bool(message.is_system_event),
            "sender_id_equals_account": bool(message.sender_is_seller),
            "raw_item_id_hash": self._hash_identifier(message.platform_item_id),
            "mapped_item_id": mapped_item_id,
            "is_item_bound": bool(is_item_bound),
            "item_bound_checked": bool(item_bound_checked),
            "item_source": item_source,
            "decision": action,
            "reason": reason,
        }
        if action == "ALLOW":
            diagnostic["allow_reason"] = (
                "account matched; sender_is_self=false (treated as buyer); "
                "listing is bound to this seller; AI enabled; session mode=AUTO"
            )
        if self.diagnostic_sink is not None:
            try:
                self.diagnostic_sink(diagnostic)
            except Exception:  # Diagnostics must never change the business decision.
                logger.warning("conversation guard diagnostic sink failed", exc_info=True)
        return ConversationDecision(action, reason)

    def check(self, message: InboundMessage) -> ConversationDecision:
        if message.account_id != self.account_id:
            return self._finish(
                message,
                action="IGNORE",
                reason="wrong_account",
                mapped_item_id=None,
                is_item_bound=False,
                item_bound_checked=False,
            )
        if message.sender_is_seller:
            return self._finish(
                message,
                action="IGNORE",
                reason="seller_echo",
                mapped_item_id=None,
                is_item_bound=False,
                item_bound_checked=False,
            )
        if message.is_system_event or message.message_type != "text" or not message.text.strip():
            return self._finish(
                message,
                action="IGNORE",
                reason="non_text_or_system",
                mapped_item_id=None,
                is_item_bound=False,
                item_bound_checked=False,
            )

        platform_item_id = str(message.platform_item_id or "").strip()
        if not platform_item_id:
            current_item_id = self.store.get_current_bound_item_id(
                self.account_id,
                message.chat_id,
            )
            if current_item_id is not None:
                return self._check_controls(
                    message,
                    mapped_item_id=current_item_id,
                    item_source="conversation_context",
                )
            return self._finish(
                message,
                action="IGNORE",
                reason="missing_platform_item_id",
                mapped_item_id=None,
                is_item_bound=False,
                item_bound_checked=False,
            )
        item_bound = self.store.is_item_bound(self.account_id, platform_item_id)
        mapped_item_id = (
            self.store.get_bound_item_id(self.account_id, platform_item_id) if item_bound else None
        )
        if not item_bound:
            return self._finish(
                message,
                action="IGNORE",
                reason="item_not_bound_to_seller",
                mapped_item_id=mapped_item_id,
                is_item_bound=False,
                item_bound_checked=True,
            )

        return self._check_controls(
            message,
            mapped_item_id=mapped_item_id,
            item_source="event",
        )

    def _check_controls(
        self,
        message: InboundMessage,
        *,
        mapped_item_id: str,
        item_source: str,
    ) -> ConversationDecision:
        account_state = self.store.account_state(self.account_id)
        if not account_state["enabled"]:
            return self._finish(
                message,
                action="IGNORE",
                reason="ai_disabled",
                mapped_item_id=mapped_item_id,
                is_item_bound=True,
                item_bound_checked=True,
                item_source=item_source,
            )
        session_state = self.store.session_state(
            self.account_id, message.chat_id, message.buyer_id
        )
        if session_state["mode"] != "AUTO":
            return self._finish(
                message,
                action="IGNORE",
                reason="human_takeover",
                mapped_item_id=mapped_item_id,
                is_item_bound=True,
                item_bound_checked=True,
                item_source=item_source,
            )
        return self._finish(
            message,
            action="ALLOW",
            reason="seller_buyer_bound_item",
            mapped_item_id=mapped_item_id,
            is_item_bound=True,
            item_bound_checked=True,
            item_source=item_source,
        )
