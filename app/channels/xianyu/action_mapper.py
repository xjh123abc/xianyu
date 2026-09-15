"""Defensive boundary mapping from the existing /chat response to channel actions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal


ChannelAction = Literal["answer", "human_handoff", "ignore"]


@dataclass(frozen=True, slots=True)
class MappedAction:
    action: ChannelAction
    text: str = ""
    reason: str = ""


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def map_chat_response(payload: Mapping[str, Any]) -> MappedAction:
    """Accept only explicit, non-empty customer-safe responses.

    The existing API has used ``reply``/``clarify``/``handoff`` while the S3
    channel contract names them ``answer``/``human_handoff``. A clarification
    means the AI cannot safely resolve the buyer turn, so it also becomes a
    handoff. Unknown or malformed model/API output is handled the same way.
    """

    action = _text(payload.get("action")).lower()
    next_step = _text(payload.get("next_step")).lower()
    answer = _text(payload.get("answer"))
    reason = _text(payload.get("reason")) or _text(payload.get("error")) or "chat_handoff"

    if action in {"ignore"}:
        return MappedAction("ignore", reason="api_requested_ignore")
    if action in {"handoff", "human_handoff"} or next_step in {"handoff", "human_handoff"}:
        return MappedAction("human_handoff", reason=reason)
    if action in {"clarify", "clarification"} or next_step in {"clarify", "clarification"}:
        return MappedAction("human_handoff", reason="buyer_question_requires_clarification")
    if action in {"reply", "answer"}:
        if answer:
            return MappedAction("answer", text=answer)
        return MappedAction("human_handoff", reason="回答为空")

    # A legacy response may omit action but still state it cannot answer.  Do
    # not let a generic error/empty answer reach the buyer.
    if payload.get("can_answer") is False:
        return MappedAction("human_handoff", reason=reason)
    return MappedAction("human_handoff", reason="/chat 返回了未知动作")
