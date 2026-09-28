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


HANDOFF_NOTICE = "客服服务暂时不可用，请稍后再试。"


class XianyuStage3Worker:
    """Contain LLM output behind durable controls and the S2 sender boundary."""

    def __init__(
        self,
        store: ChannelStore,
        *,
        account_id: str,
        chat_client: ChatClient,
        diagnostic_sink: Callable[[Mapping[str, object]], None] | None = None,
    ) -> None:
        self.store = store
        self.account_id = account_id
        self.chat_client = chat_client
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

    async def mark_delivery_unconfirmed(self, message_id: str) -> None:
        """Record that a local send had no matching seller echo before timeout."""
        self.store.mark_delivery(
            self.account_id, message_id, "UNKNOWN", error="seller_echo_timeout"
        )
        await self._report_delivery(message_id)

    async def process(
        self,
        message: InboundMessage,
        sender: TextSender,
        *,
        already_recorded: bool = False,
    ) -> dict[str, Any]:
        if message.account_id != self.account_id:
            return {"action": "ignored", "reason": "wrong_account"}
        await self._retry_delivery_reports()
        if already_recorded:
            existing = self.store.message(self.account_id, message.platform_message_id)
            if existing is None:
                return {"action": "ignored", "reason": "recorded_message_missing"}
            if existing.get("status") == "CONTEXT_PENDING":
                return await self._record_human_context(message)
            if existing.get("status") != "RECEIVED":
                return {"action": "duplicate", "message_id": message.platform_message_id}
        elif not self.store.record_inbound(message):
            existing = self.store.message(self.account_id, message.platform_message_id)
            if existing is not None and existing.get("status") == "CONTEXT_PENDING":
                return await self._record_human_context(message)
            return {"action": "duplicate", "message_id": message.platform_message_id}

        if message.sender_is_seller:
            confirmed_message_id = self.store.confirm_delivery_echo(
                self.account_id, message.chat_id, message.text
            )
            if confirmed_message_id is not None:
                self.store.mark_ignored(
                    self.account_id, message.platform_message_id, "seller_delivery_echo"
                )
                await self._report_delivery(confirmed_message_id)
                return {
                    "action": "delivery_confirmation",
                    "delivery": "confirmed",
                    "message_id": confirmed_message_id,
                }

        session = self.store.get_session_state(self.account_id, message.chat_id)
        if session is not None and session.get("mode") == "HUMAN":
            return await self._record_human_context(message)

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
            self._record_delivery_reference(message.platform_message_id, response)
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

    async def recover_after_restart(self, sender: TextSender) -> None:
        """Retry receipts and only resume candidates that were never claimed for send."""

        await self._retry_delivery_reports()
        for pending in self.store.pending_ready_deliveries(self.account_id):
            message_id = str(pending["platform_message_id"])
            delivery = self.store.claim_ready_delivery(self.account_id, message_id)
            if delivery is not None:
                await self._send(message_id, delivery, sender, str(delivery["action"]))

    async def _record_human_context(self, message: InboundMessage) -> dict[str, Any]:
        """Store buyer/seller text during takeover, without calling /chat generation."""
        if message.is_system_event or message.message_type != "text" or not message.text.strip():
            self.store.mark_ignored(self.account_id, message.platform_message_id, "human_non_text_event")
            return {"action": "ignored", "reason": "human_non_text_event"}
        if message.sender_is_seller and self.store.is_delivery_echo(
            self.account_id, message.chat_id, message.text
        ):
            self.store.mark_ignored(self.account_id, message.platform_message_id, "seller_delivery_echo")
            return {"action": "ignored", "reason": "seller_delivery_echo"}

        self.store.mark_human_context_pending(self.account_id, message.platform_message_id)
        try:
            recorder = getattr(self.chat_client, "record_conversation_event", None)
            if not callable(recorder):
                raise RuntimeError("chat client does not support conversation events")
            result = await recorder(
                account_id=self.account_id,
                chat_id=message.chat_id,
                event_id=message.platform_message_id,
                role="assistant" if message.sender_is_seller else "user",
                source="seller_manual" if message.sender_is_seller else "buyer_message",
                content=message.text,
            )
        except Exception as error:
            return {"action": "context_record_failed", "reason": type(error).__name__}
        self.store.mark_ignored(self.account_id, message.platform_message_id, "human_context_recorded")
        return {"action": "context_recorded", "appended": bool(result.get("appended"))}

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
            await self._report_delivery(message_id)
            return {"action": action, "delivery": "unknown", "request_id": delivery["request_id"]}
        except (ConnectionError, OSError):
            self.store.mark_delivery(self.account_id, message_id, "FAILED", error="send_connection_failed")
            await self._report_delivery(message_id)
            return {"action": action, "delivery": "failed", "request_id": delivery["request_id"]}
        except Exception:
            self.store.mark_delivery(self.account_id, message_id, "FAILED", error="send_failed")
            await self._report_delivery(message_id)
            return {"action": action, "delivery": "failed", "request_id": delivery["request_id"]}
        if not isinstance(receipt, SendReceipt) or not receipt.local_submitted:
            self.store.mark_delivery(self.account_id, message_id, "FAILED", error="sender_did_not_submit")
            await self._report_delivery(message_id)
            return {"action": action, "delivery": "failed", "request_id": delivery["request_id"]}
        state = "CONFIRMED" if receipt.platform_confirmed else "LOCAL_SUBMITTED"
        self.store.mark_delivery(self.account_id, message_id, state)
        await self._report_delivery(message_id)
        return {"action": action, "delivery": state.lower(), "request_id": delivery["request_id"]}

    def _record_delivery_reference(
        self, message_id: str, response: Mapping[str, Any]
    ) -> None:
        turn_id = response.get("turn_id")
        proposal_id = response.get("proposal_id")
        if isinstance(turn_id, str) and turn_id.strip() and isinstance(proposal_id, str) and proposal_id.strip():
            self.store.record_delivery_reference(
                self.account_id,
                message_id,
                turn_id=turn_id,
                proposal_id=proposal_id,
            )

    async def _retry_delivery_reports(self) -> None:
        for delivery in self.store.pending_delivery_reports(self.account_id):
            await self._report_delivery(str(delivery["platform_message_id"]))

    async def _report_delivery(self, message_id: str) -> None:
        """Retry a state receipt only; this method never sends buyer text."""

        reporter = getattr(self.chat_client, "report_delivery", None)
        if not callable(reporter):
            return
        pending = [
            row
            for row in self.store.pending_delivery_reports(self.account_id)
            if row["platform_message_id"] == message_id
        ]
        if not pending:
            return
        receipt = pending[0]
        try:
            result = await reporter(
                chat_id=str(receipt["chat_id"]),
                turn_id=str(receipt["turn_id"]),
                proposal_id=str(receipt["proposal_id"]),
                delivery_state=str(receipt["delivery_state"]),
            )
        except Exception:
            self.store.mark_delivery_report(
                self.account_id, message_id, "RETRY", error="delivery_receipt_failed"
            )
            return
        if isinstance(result, Mapping) and result.get("accepted") is True:
            self.store.mark_delivery_report(self.account_id, message_id, "SENT")
        else:
            self.store.mark_delivery_report(
                self.account_id, message_id, "REJECTED", error="delivery_receipt_rejected"
            )
