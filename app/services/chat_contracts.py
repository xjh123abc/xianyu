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
ExecutionMode = Literal[
    "xianyu_expert",
    "general_rag",
    "order",
    "missing_order_id",
    "common_knowledge",
    "unsupported_action",
]
TASK_TYPES = frozenset({"product", "price", "service", "order"})
EXECUTION_MODES = frozenset(
    {
        "xianyu_expert",
        "general_rag",
        "order",
        "missing_order_id",
        "common_knowledge",
        "unsupported_action",
    }
)


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
    """Return fresh, item-scoped state for the price-negotiation flow."""

    return {
        "item_id": None,
        "round": 0,
        "last_ai_offer": None,
        "last_buyer_offer": None,
        "shipping_condition": None,
        "policy_version": None,
        "offer_status": "none",
        "pending_offer": None,
        "last_committed_turn_id": None,
        "last_proposal_id": None,
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
    # Channel adapters provide a stable value derived from the platform
    # message identifier.  Direct /chat callers may leave it absent.
    turn_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "platform", _required_text(self.platform, "platform"))
        object.__setattr__(self, "account_id", _required_text(self.account_id, "account_id"))
        object.__setattr__(self, "chat_id", _required_text(self.chat_id, "chat_id"))
        object.__setattr__(self, "buyer_id", _required_text(self.buyer_id, "buyer_id"))
        object.__setattr__(self, "item_id", _optional_text(self.item_id, "item_id"))
        object.__setattr__(self, "text", _required_text(self.text, "text"))
        object.__setattr__(self, "turn_id", _optional_text(self.turn_id, "turn_id"))


@dataclass(frozen=True, slots=True)
class SessionContext:
    """The session information future planners may read for one buyer turn."""

    history: list[dict[str, str]] = field(default_factory=list)
    current_item_id: str | None = None
    current_order_id: str | None = None
    last_task_type: TaskType | None = None
    negotiation: dict[str, object] = field(default_factory=default_negotiation_state)
    platform_context: dict[str, dict[str, object]] = field(default_factory=dict)
    deadline: float | None = None

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
        for field_name in ("last_ai_offer", "last_buyer_offer"):
            amount = state[field_name]
            if amount is not None and (
                not isinstance(amount, int) or isinstance(amount, bool) or amount < 0
            ):
                raise ValueError(f"negotiation.{field_name} must be non-negative cents or None")
        if state["shipping_condition"] not in {None, "seller_pays", "buyer_pays"}:
            raise ValueError("negotiation.shipping_condition is invalid")
        if state["policy_version"] is not None and (
            not isinstance(state["policy_version"], int)
            or isinstance(state["policy_version"], bool)
            or state["policy_version"] < 0
        ):
            raise ValueError("negotiation.policy_version must be a non-negative integer or None")
        if state["offer_status"] not in {
            "none",
            "generated",
            "submitted",
            "confirmed",
            "failed",
            "unknown",
        }:
            raise ValueError("negotiation.offer_status is invalid")
        for field_name in ("last_committed_turn_id", "last_proposal_id"):
            state[field_name] = _optional_text(state[field_name], f"negotiation.{field_name}")
        if state["pending_offer"] is not None and not isinstance(state["pending_offer"], Mapping):
            raise ValueError("negotiation.pending_offer must be a mapping or None")
        if isinstance(state["pending_offer"], Mapping):
            state["pending_offer"] = dict(state["pending_offer"])
        object.__setattr__(self, "negotiation", state)
        if not isinstance(self.platform_context, Mapping) or any(
            not isinstance(name, str) or not isinstance(value, Mapping)
            for name, value in self.platform_context.items()
        ):
            raise ValueError("platform_context must map platform names to mappings")
        object.__setattr__(
            self,
            "platform_context",
            {name: dict(value) for name, value in self.platform_context.items()},
        )


@dataclass(frozen=True, slots=True)
class Task:
    """One planner-owned unit of work for a business handler.

    Routing and scheduling fields are explicit. ``metadata`` is reserved for
    capability-specific parameters and temporary planner compatibility state.
    """

    task_id: str
    task_type: TaskType
    query: str
    metadata: dict[str, object] = field(default_factory=dict)
    query_target: str | None = None
    depends_on_task_ids: tuple[str, ...] = ()
    execution_mode: ExecutionMode | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _required_text(self.task_id, "task_id"))
        if self.task_type not in TASK_TYPES:
            raise ValueError("task_type must be product, price, service, or order")
        object.__setattr__(self, "query", _required_text(self.query, "query"))
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(
            self,
            "query_target",
            _optional_text(self.query_target, "query_target"),
        )
        dependencies = self.depends_on_task_ids
        if not isinstance(dependencies, (list, tuple)) or any(
            not isinstance(dependency, str) or not dependency.strip()
            for dependency in dependencies
        ):
            raise ValueError("depends_on_task_ids must contain non-empty strings")
        normalized_dependencies = tuple(dependency.strip() for dependency in dependencies)
        if len(set(normalized_dependencies)) != len(normalized_dependencies):
            raise ValueError("depends_on_task_ids must not contain duplicates")
        if self.task_id in normalized_dependencies:
            raise ValueError("task cannot depend on itself")
        object.__setattr__(self, "depends_on_task_ids", normalized_dependencies)
        if self.execution_mode is not None and self.execution_mode not in EXECUTION_MODES:
            raise ValueError("execution_mode is invalid")


@dataclass(frozen=True, slots=True)
class TaskResult:
    """One task outcome, including an explicit reason when it is unavailable."""

    task_id: str
    status: str
    answer: str
    sources: list[dict[str, object]] = field(default_factory=list)
    reason: str | None = None
    metadata: dict[str, object] = field(default_factory=dict, compare=False)

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
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        object.__setattr__(self, "metadata", dict(self.metadata))


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
