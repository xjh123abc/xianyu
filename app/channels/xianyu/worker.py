"""S2 fixed-reply worker with durable controls and delivery states."""

from __future__ import annotations

from typing import Any

from app.channels.xianyu.adapter import iter_sync_events
from app.channels.xianyu.client import TextSender
from app.channels.xianyu.models import InboundMessage, SendReceipt
from app.channels.xianyu.store import ChannelStore


class XianyuStage2Worker:
    """Process only an explicitly configured test message in S2."""

    def __init__(
        self,
        store: ChannelStore,
        *,
        account_id: str,
        trigger_text: str,
        fixed_reply: str = "闲鱼AI客服接通测试成功",
    ) -> None:
        if not account_id.strip() or not trigger_text.strip() or not fixed_reply.strip():
            raise ValueError("account, trigger and fixed reply are required")
        self.store = store
        self.account_id = account_id
        self.trigger_text = trigger_text.strip()
        self.fixed_reply = fixed_reply.strip()

    def enable_account(self) -> int:
        return self.store.set_enabled(self.account_id, True)

    def pause_account(self) -> int:
        return self.store.set_enabled(self.account_id, False)

    def takeover(self, chat_id: str, buyer_id: str, reason: str = "seller_takeover") -> int:
        return self.store.set_session_mode(self.account_id, chat_id, buyer_id, "HUMAN", reason)

    def release(self, chat_id: str, buyer_id: str) -> int:
        return self.store.set_session_mode(self.account_id, chat_id, buyer_id, "AUTO", None)

    async def process(self, message: InboundMessage, sender: TextSender) -> dict[str, Any]:
        if message.account_id != self.account_id:
            return {"action": "ignored", "reason": "wrong_account"}
        inserted = self.store.record_inbound(message)
        if not inserted:
            return {"action": "duplicate", "message_id": message.platform_message_id}
        if message.sender_is_seller:
            self.store.mark_ignored(self.account_id, message.platform_message_id, "seller_echo")
            return {"action": "ignored", "reason": "seller_echo"}
        if message.is_system_event or message.message_type != "text":
            self.store.mark_ignored(self.account_id, message.platform_message_id, "non_text_or_system")
            return {"action": "ignored", "reason": "non_text_or_system"}
        if message.text.strip() != self.trigger_text:
            self.store.mark_ignored(self.account_id, message.platform_message_id, "not_test_trigger")
            return {"action": "ignored", "reason": "not_test_trigger"}

        claim = self.store.claim_for_send(
            self.account_id,
            message.platform_message_id,
            self.fixed_reply,
        )
        if claim is None:
            return {"action": "blocked", "reason": "account_paused_or_human_or_already_claimed"}
        try:
            receipt = await sender.send_text(
                claim["chat_id"],
                claim["buyer_id"],
                self.fixed_reply,
                claim["request_id"],
            )
        except TimeoutError as exc:
            self.store.mark_delivery(self.account_id, message.platform_message_id, "UNKNOWN", error=str(exc))
            return {"action": "unknown", "request_id": claim["request_id"]}
        except (ConnectionError, OSError) as exc:
            self.store.mark_delivery(self.account_id, message.platform_message_id, "FAILED", error=str(exc))
            return {"action": "failed", "request_id": claim["request_id"]}
        except Exception as exc:
            self.store.mark_delivery(self.account_id, message.platform_message_id, "FAILED", error=type(exc).__name__)
            return {"action": "failed", "request_id": claim["request_id"]}

        if not isinstance(receipt, SendReceipt) or not receipt.local_submitted:
            self.store.mark_delivery(self.account_id, message.platform_message_id, "FAILED", error="sender did not submit")
            return {"action": "failed", "request_id": claim["request_id"]}
        if receipt.platform_confirmed:
            self.store.mark_delivery(self.account_id, message.platform_message_id, "CONFIRMED")
            action = "confirmed"
        else:
            self.store.mark_delivery(self.account_id, message.platform_message_id, "LOCAL_SUBMITTED")
            action = "submitted"
        return {"action": action, "request_id": claim["request_id"]}

    async def process_sync_message(
        self,
        payload: dict[str, Any],
        sender: TextSender,
        *,
        seller_id: str,
        decrypt: Any,
    ) -> list[dict[str, Any]]:
        """Process every event in one sync package through the same ledger."""

        results: list[dict[str, Any]] = []
        for event in iter_sync_events(
            payload,
            account_id=self.account_id,
            seller_id=seller_id,
            decrypt=decrypt,
        ):
            results.append(await self.process(event, sender))
        return results
