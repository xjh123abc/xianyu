"""S3 buyer-message to /chat to Xianyu delivery worker."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from app.channels.xianyu.action_mapper import map_chat_response
from app.channels.xianyu.adapter import to_chat_message
from app.channels.xianyu.chat_client import ChatClient
from app.channels.xianyu.client import TextSender
from app.channels.xianyu.conversation_guard import ConversationGuard
from app.channels.xianyu.models import InboundMessage, SendReceipt
from app.channels.xianyu.store import ChannelStore
from app.channels.xianyu.wecom import HandoffNotifier


HANDOFF_NOTICE = "客服服务暂时不可用，请稍后再试。"


class XianyuStage3Worker:
    """Contain LLM output behind durable controls and the S2 sender boundary."""

    def __init__(
        self,
        store: ChannelStore,
        *,
        account_id: str,
        chat_client: ChatClient,
        notifier: HandoffNotifier,
        diagnostic_sink: Callable[[Mapping[str, object]], None] | None = None,
    ) -> None:
        self.store = store
        self.account_id = account_id
        self.chat_client = chat_client
        self.notifier = notifier
        self.conversation_guard = ConversationGuard(
            store,
            account_id=account_id,
            diagnostic_sink=diagnostic_sink,
        )

    def enable_account(self) -> int:
        return self.store.set_enabled(self.account_id, True)

    def pause_account(self) -> int:
        return self.store.set_enabled(self.account_id, False)

    def takeover(self, chat_id: str, buyer_id: str, reason: str = "seller_takeover") -> int:
        return self.store.set_session_mode(self.account_id, chat_id, buyer_id, "HUMAN", reason)

    def release(self, chat_id: str, buyer_id: str) -> int:
        return self.store.set_session_mode(self.account_id, chat_id, buyer_id, "AUTO")

    async def process(self, message: InboundMessage, sender: TextSender) -> dict[str, Any]:
        if message.account_id != self.account_id:
            return {"action": "ignored", "reason": "wrong_account"}
        if not self.store.record_inbound(message):
            return {"action": "duplicate", "message_id": message.platform_message_id}

        guard = self.conversation_guard.check(message)
        if guard.action == "IGNORE":
            self.store.mark_ignored(self.account_id, message.platform_message_id, guard.reason)
            return {"action": "ignored", "reason": guard.reason}

        item_id = self.store.resolve_item(
            self.account_id, message.chat_id, message.buyer_id, message.platform_item_id
        )
        if item_id is None:
            # The binding can be removed after the guard check.  Never call
            # /chat for a listing whose ownership can no longer be verified.
            self.store.mark_ignored(
                self.account_id,
                message.platform_message_id,
                "item_not_bound_to_seller",
            )
            return {"action": "ignored", "reason": "item_not_bound_to_seller"}
        claim = self.store.claim_for_generation(self.account_id, message.platform_message_id)
        if claim is None:
            return {"action": "blocked", "reason": "account_paused_human_or_claimed"}

        try:
            chat_message = to_chat_message(message, item_id=item_id)
            response = await self.chat_client.ask(chat_message)
            decision = map_chat_response(response)
        except (TimeoutError, ConnectionError, OSError):
            return await self._auto_error(message, sender, claim, "客服服务暂时不可用")
        except Exception:
            # Do not surface stack traces, endpoint details, or model errors to a buyer.
            return await self._auto_error(message, sender, claim, "chat_service_unexpected_error")

        if decision.action == "ignore":
            self.store.mark_ignored(self.account_id, message.platform_message_id, decision.reason)
            return {"action": "ignored", "reason": decision.reason}
        if decision.action == "error":
            return await self._auto_error(message, sender, claim, decision.reason)

        prepared = self.store.prepare_candidate(
            self.account_id,
            message.platform_message_id,
            action=decision.action,
            candidate_text=decision.text,
            account_version=claim["account_version"],
            session_version=claim["session_version"],
        )
        if prepared is None:
            return {"action": "superseded", "reason": "control_changed_during_generation"}
        delivery = self.store.claim_ready_delivery(self.account_id, message.platform_message_id)
        if delivery is None:
            return {"action": "blocked", "reason": "account_paused_or_human_before_send"}
        return await self._send(message.platform_message_id, delivery, sender, decision.action)

    async def _auto_error(
        self,
        message: InboundMessage,
        sender: TextSender,
        claim: Mapping[str, object],
        reason: str,
    ) -> dict[str, Any]:
        """Send a fixed safe notice while preserving AUTO session ownership."""

        prepared = self.store.prepare_candidate(
            self.account_id,
            message.platform_message_id,
            action="clarify",
            candidate_text=HANDOFF_NOTICE,
            account_version=int(claim["account_version"]),
            session_version=int(claim["session_version"]),
        )
        if prepared is None:
            return {"action": "superseded", "reason": "control_changed_during_generation"}
        delivery = self.store.claim_ready_delivery(self.account_id, message.platform_message_id)
        if delivery is None:
            return {"action": "blocked", "reason": "account_paused_or_human_before_send"}
        result = await self._send(message.platform_message_id, delivery, sender, "error")
        result["reason"] = reason
        return result

    async def _send(
        self, message_id: str, delivery: dict[str, Any], sender: TextSender, action: str
    ) -> dict[str, Any]:
        try:
            receipt = await sender.send_text(
                delivery.get("platform_chat_id") or delivery["chat_id"],
                delivery["buyer_id"],
                delivery["candidate_text"],
                delivery["request_id"],
            )
        except TimeoutError:
            self.store.mark_delivery(self.account_id, message_id, "UNKNOWN", error="send_timeout")
            return {"action": action, "delivery": "unknown", "request_id": delivery["request_id"]}
        except (ConnectionError, OSError):
            self.store.mark_delivery(self.account_id, message_id, "FAILED", error="send_connection_failed")
            return {"action": action, "delivery": "failed", "request_id": delivery["request_id"]}
        except Exception:
            self.store.mark_delivery(self.account_id, message_id, "FAILED", error="send_failed")
            return {"action": action, "delivery": "failed", "request_id": delivery["request_id"]}
        if not isinstance(receipt, SendReceipt) or not receipt.local_submitted:
            self.store.mark_delivery(self.account_id, message_id, "FAILED", error="sender_did_not_submit")
            return {"action": action, "delivery": "failed", "request_id": delivery["request_id"]}
        state = "CONFIRMED" if receipt.platform_confirmed else "LOCAL_SUBMITTED"
        self.store.mark_delivery(self.account_id, message_id, state)
        return {"action": action, "delivery": state.lower(), "request_id": delivery["request_id"]}
