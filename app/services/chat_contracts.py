"""Transport-independent contracts for the gradual chat refactor.

These types describe the boundary between channel adapters and the existing
chat core. They intentionally do not import a channel, router, or storage
implementation so later steps can depend on one stable vocabulary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal


TaskType = Literal["product", "price", "service", "order"]
TASK_TYPES = frozenset({"product", "price", "service", "order"})


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not (normalized := value.strip()):
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field_name)


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value.strip()


def default_negotiation_state() -> dict[str, object]:
    """Return fresh state reserved for the future price-negotiation flow."""

    return {
        "item_id": None,
        "round": 0,
        "last_ai_offer": None,
        "last_buyer_offer": None,
    }


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One buyer text turn after a platform adapter has normalized it."""

    platform: str
    account_id: str
    chat_id: str
    buyer_id: str
    item_id: str | None
    text: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "platform", _required_text(self.platform, "platform"))
        object.__setattr__(self, "account_id", _required_text(self.account_id, "account_id"))
        object.__setattr__(self, "chat_id", _required_text(self.chat_id, "chat_id"))
        object.__setattr__(self, "buyer_id", _required_text(self.buyer_id, "buyer_id"))
        object.__setattr__(self, "item_id", _optional_text(self.item_id, "item_id"))
        object.__setattr__(self, "text", _required_text(self.text, "text"))


@dataclass(frozen=True, slots=True)
class SessionContext:
    """The session information future planners may read for one buyer turn."""

    history: list[dict[str, str]] = field(default_factory=list)
    current_item_id: str | None = None
    current_order_id: str | None = None
    last_task_type: TaskType | None = None
    negotiation: dict[str, object] = field(default_factory=default_negotiation_state)

    def __post_init__(self) -> None:
        if not isinstance(self.history, list) or any(
            not isinstance(turn, Mapping) for turn in self.history
        ):
            raise ValueError("history must be a list of mappings")
        object.__setattr__(self, "history", [dict(turn) for turn in self.history])
        object.__setattr__(
            self,
            "current_item_id",
            _optional_text(self.current_item_id, "current_item_id"),
        )
        object.__setattr__(
            self,
            "current_order_id",
            _optional_text(self.current_order_id, "current_order_id"),
        )
        if self.last_task_type is not None and self.last_task_type not in TASK_TYPES:
            raise ValueError("last_task_type must be product, price, service, or order")
        if not isinstance(self.negotiation, Mapping):
            raise ValueError("negotiation must be a mapping")
        state = default_negotiation_state()
        state.update(dict(self.negotiation))
        round_number = state["round"]
        if not isinstance(round_number, int) or isinstance(round_number, bool) or round_number < 0:
            raise ValueError("negotiation.round must be a non-negative integer")
        state["item_id"] = _optional_text(state["item_id"], "negotiation.item_id")
        object.__setattr__(self, "negotiation", state)


@dataclass(frozen=True, slots=True)
class Task:
    """One planner-owned unit of work for a business handler."""

    task_id: str
    task_type: TaskType
    query: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _required_text(self.task_id, "task_id"))
        if self.task_type not in TASK_TYPES:
            raise ValueError("task_type must be product, price, service, or order")
        object.__setattr__(self, "query", _required_text(self.query, "query"))


@dataclass(frozen=True, slots=True)
class TaskResult:
    """One task outcome, including an explicit reason when it is unavailable."""

    task_id: str
    status: str
    answer: str
    sources: list[dict[str, object]] = field(default_factory=list)
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _required_text(self.task_id, "task_id"))
        object.__setattr__(self, "status", _required_text(self.status, "status"))
        object.__setattr__(self, "answer", _text(self.answer, "answer"))
        if not isinstance(self.sources, list) or any(
            not isinstance(source, Mapping) for source in self.sources
        ):
            raise ValueError("sources must be a list of mappings")
        object.__setattr__(self, "sources", [dict(source) for source in self.sources])
        object.__setattr__(self, "reason", _optional_text(self.reason, "reason"))


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """Future unified output while legacy API response fields remain compatible."""

    action: str
    answer: str
    results: list[TaskResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", _required_text(self.action, "action"))
        object.__setattr__(self, "answer", _text(self.answer, "answer"))
        if not isinstance(self.results, list) or any(
            not isinstance(result, TaskResult) for result in self.results
        ):
            raise ValueError("results must be a list of TaskResult values")
        object.__setattr__(self, "results", list(self.results))
