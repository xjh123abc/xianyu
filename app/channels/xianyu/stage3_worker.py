"""S3 buyer-message to /chat to Xianyu delivery worker."""

from __future__ import annotations

from typing import Any

from app.channels.xianyu.action_mapper import map_chat_response
from app.channels.xianyu.chat_client import ChatClient
from app.channels.xianyu.client import TextSender
from app.channels.xianyu.models import InboundMessage, SendReceipt
from app.channels.xianyu.store import ChannelStore
from app.channels.xianyu.wecom import HandoffNotifier


HANDOFF_NOTICE = "这个问题需要卖家确认，已记录，卖家看到后会处理。"


class XianyuStage3Worker:
    """Contain LLM output behind durable controls and the S2 sender boundary."""

    def __init__(
        self,
        store: ChannelStore,
        *,
        account_id: str,
        chat_client: ChatClient,
        notifier: HandoffNotifier,
    ) -> None:
        self.store = store
        self.account_id = account_id
        self.chat_client = chat_client
        self.notifier = notifier

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
        if message.sender_is_seller:
            self.store.mark_ignored(self.account_id, message.platform_message_id, "seller_echo")
            return {"action": "ignored", "reason": "seller_echo"}
        if message.is_system_event or message.message_type != "text" or not message.text.strip():
            self.store.mark_ignored(self.account_id, message.platform_message_id, "non_text_or_system")
            return {"action": "ignored", "reason": "non_text_or_system"}

        item_id = self.store.resolve_item(
            self.account_id, message.chat_id, message.buyer_id, message.platform_item_id
        )
        claim = self.store.claim_for_generation(self.account_id, message.platform_message_id)
        if claim is None:
            return {"action": "blocked", "reason": "account_paused_human_or_claimed"}

        try:
            response = await self.chat_client.ask(
                query=message.text.strip(), chat_id=message.chat_id, item_id=item_id
            )
            decision = map_chat_response(response)
        except (TimeoutError, ConnectionError, OSError):
            return await self._handoff(message, sender, item_id, "客服服务暂时不可用")
        except Exception:
            # Do not surface stack traces, endpoint details, or model errors to a buyer.
            return await self._handoff(message, sender, item_id, "客服回答需要卖家确认")

        if decision.action == "ignore":
            self.store.mark_ignored(self.account_id, message.platform_message_id, decision.reason)
            return {"action": "ignored", "reason": decision.reason}
        if decision.action == "human_handoff":
            return await self._handoff(message, sender, item_id, decision.reason)

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

    async def _handoff(
        self, message: InboundMessage, sender: TextSender, item_id: str | None, reason: str
    ) -> dict[str, Any]:
        handoff = self.store.handoff(
            self.account_id,
            message.platform_message_id,
            reason=reason,
            candidate_text=HANDOFF_NOTICE,
        )
        if handoff is None:
            return {"action": "superseded", "reason": "account_paused_or_human"}
        try:
            await self.notifier.notify_handoff(
                chat_id=handoff["chat_id"], item_id=item_id, reason=reason, question=handoff["text"]
            )
        except Exception:
            self.store.mark_notification(self.account_id, message.platform_message_id, "FAILED", "notification_failed")
        else:
            self.store.mark_notification(self.account_id, message.platform_message_id, "SENT")

        # HUMAN was already committed.  The optional buyer notice is therefore
        # a one-time controlled send, never another model-generated response.
        delivery = self.store.claim_handoff_notice(self.account_id, message.platform_message_id)
        if delivery is None:
            return {"action": "human_handoff", "delivery": "not_sent"}
        result = await self._send(
            message.platform_message_id,
            delivery,
            sender,
            "human_handoff",
        )
        result["action"] = "human_handoff"
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
