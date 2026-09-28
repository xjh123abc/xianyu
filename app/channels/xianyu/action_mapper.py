"""Defensive boundary mapping from the existing /chat response to channel actions."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal


ChannelAction = Literal["answer", "clarify", "ignore", "error"]


@dataclass(frozen=True, slots=True)
class MappedAction:
    action: ChannelAction
    text: str = ""
    reason: str = ""


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def map_chat_response(payload: Mapping[str, Any]) -> MappedAction:
    """Accept only explicit, non-empty customer-safe responses.

    Clarification and failed-answer states must not seize a seller's session.
    Only an explicit seller takeover is allowed to change a session to HUMAN.
    """

    action = _text(payload.get("action")).lower()
    next_step = _text(payload.get("next_step")).lower()
    answer = _text(payload.get("answer"))
    source_reason = _text(payload.get("reason")) or _text(payload.get("error"))
    reason = source_reason or "chat_handoff"

    if action in {"ignore"}:
        return MappedAction("ignore", reason="api_requested_ignore")
    if action in {"handoff", "human_handoff"} or next_step in {"handoff", "human_handoff"}:
        # Old API payloads may still use handoff.  The channel no longer
        # replies or takes over for that legacy action; explicit seller
        # controls remain the only path into HUMAN mode.
        return MappedAction("ignore", reason=reason)
    if action in {"clarify", "clarification"} or next_step in {"clarify", "clarification"}:
        return MappedAction(
            "clarify",
            text=answer or "请补充一下商品或订单信息。",
            reason="buyer_question_requires_clarification",
        )
    if action in {"reply", "answer"}:
        if answer:
            return MappedAction("answer", text=answer)
        # Keep a stable upstream failure code in channel logs so an empty
        # response can be traced to the expert that produced it.
        reason_code = source_reason.split(":", 1)[0] if source_reason else "empty_answer"
        if not re.fullmatch(r"[a-zA-Z0-9_.-]{1,80}", reason_code):
            reason_code = "empty_answer"
        return MappedAction("error", reason=reason_code)

    # A legacy response may omit action but still carry a model answer.  Never
    # substitute a fixed unavailable reply for that text.
    if payload.get("can_answer") is False:
        return MappedAction("answer", text=answer, reason=reason) if answer else MappedAction("error", reason=reason)
    return MappedAction("error", reason="unknown_chat_action")
