"""Small, transport-independent types for the Xianyu channel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


MessageType = Literal["text", "system", "attachment", "unknown"]
SessionMode = Literal["AUTO", "HUMAN"]
DeliveryState = Literal["NONE", "LOCAL_SUBMITTED", "CONFIRMED", "FAILED", "UNKNOWN"]


@dataclass(frozen=True, slots=True)
class InboundMessage:
    """A normalized event; IDs are supplied by the trusted channel adapter."""

    account_id: str
    platform_message_id: str
    chat_id: str
    buyer_id: str
    text: str
    message_type: MessageType = "text"
    platform_item_id: str | None = None
    sender_is_seller: bool = False
    is_system_event: bool = False
    received_at: str | None = None


@dataclass(frozen=True, slots=True)
class SendReceipt:
    """Evidence returned by a sender after its local submit attempt."""

    request_id: str
    local_submitted: bool
    platform_confirmed: bool = False
    self_echo_observed: bool = False
    error: str | None = None
