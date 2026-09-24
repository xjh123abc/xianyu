"""Semantic intent contracts for one buyer turn."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal


UnderstandingStatus = Literal["ready", "clarify", "error"]


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not (normalized := value.strip()):
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field_name)


def _mapping_tuple(
    value: Sequence[Mapping[str, object]] | None,
    field_name: str,
) -> tuple[Mapping[str, object], ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(entry, Mapping) for entry in value
    ):
        raise ValueError(f"{field_name} must be a sequence of mappings")
    return tuple(dict(entry) for entry in value)


def _text_tuple(value: Sequence[str] | None, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(entry, str) or not entry.strip() for entry in value
    ):
        raise ValueError(f"{field_name} must be a sequence of non-empty strings")
    return tuple(entry.strip() for entry in value)


@dataclass(frozen=True, slots=True)
class NeedDependency:
    """A semantic relation between two buyer needs."""

    need_id: str
    depends_on_need_id: str
    relation: str = "precondition"
    condition: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "need_id", _required_text(self.need_id, "need_id"))
        object.__setattr__(
            self,
            "depends_on_need_id",
            _required_text(self.depends_on_need_id, "depends_on_need_id"),
        )
        object.__setattr__(self, "relation", _required_text(self.relation, "relation"))
        if not isinstance(self.condition, Mapping):
            raise ValueError("condition must be a mapping")
        object.__setattr__(self, "condition", dict(self.condition))


@dataclass(frozen=True, slots=True)
class UserNeed:
    """One independently trackable semantic request in the current buyer turn."""

    need_id: str
    intent: str
    original_question: str
    normalized_question: str
    subject: str | None = None
    conditions: tuple[Mapping[str, object], ...] = ()
    source_texts: tuple[str, ...] = ()
    context_references: tuple[Mapping[str, object], ...] = ()
    requested_outcome: str | None = None
    reply_required: bool = True
    depends_on_need_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "need_id", _required_text(self.need_id, "need_id"))
        object.__setattr__(self, "intent", _required_text(self.intent, "intent"))
        object.__setattr__(
            self,
            "original_question",
            _required_text(self.original_question, "original_question"),
        )
        object.__setattr__(
            self,
            "normalized_question",
            _required_text(self.normalized_question, "normalized_question"),
        )
        object.__setattr__(self, "subject", _optional_text(self.subject, "subject"))
        object.__setattr__(
            self,
            "conditions",
            _mapping_tuple(self.conditions, "conditions"),
        )
        object.__setattr__(
            self,
            "source_texts",
            _text_tuple(self.source_texts or (self.original_question,), "source_texts"),
        )
        object.__setattr__(
            self,
            "context_references",
            _mapping_tuple(self.context_references, "context_references"),
        )
        object.__setattr__(
            self,
            "requested_outcome",
            _optional_text(self.requested_outcome, "requested_outcome"),
        )
        if not isinstance(self.reply_required, bool):
            raise ValueError("reply_required must be a bool")
        dependencies = _text_tuple(self.depends_on_need_ids, "depends_on_need_ids")
        if self.need_id in dependencies:
            raise ValueError("a need cannot depend on itself")
        object.__setattr__(self, "depends_on_need_ids", dependencies)

    def intent_context(self) -> dict[str, object]:
        """Return a JSON-safe semantic payload for Task metadata."""

        return {
            "need_id": self.need_id,
            "intent": self.intent,
            "subject": self.subject,
            "conditions": [dict(condition) for condition in self.conditions],
            "source_texts": list(self.source_texts),
            "context_references": [
                dict(reference) for reference in self.context_references
            ],
            "requested_outcome": self.requested_outcome,
            "reply_required": self.reply_required,
            "depends_on_need_ids": list(self.depends_on_need_ids),
        }


@dataclass(frozen=True, slots=True)
class UnderstandingResult:
    """The semantic understanding result before deterministic task mapping."""

    status: UnderstandingStatus
    needs: tuple[UserNeed, ...] = ()
    clarification_question: str | None = None
    error_reason: str | None = None
    dependencies: tuple[NeedDependency, ...] = ()
    raw_response: object | None = None
    model_called: bool = False

    def __post_init__(self) -> None:
        if self.status not in {"ready", "clarify", "error"}:
            raise ValueError("status must be ready, clarify, or error")
        if not isinstance(self.needs, (list, tuple)) or any(
            not isinstance(need, UserNeed) for need in self.needs
        ):
            raise ValueError("needs must contain UserNeed values")
        object.__setattr__(self, "needs", tuple(self.needs))
        object.__setattr__(
            self,
            "clarification_question",
            _optional_text(self.clarification_question, "clarification_question"),
        )
        object.__setattr__(
            self,
            "error_reason",
            _optional_text(self.error_reason, "error_reason"),
        )
        if not isinstance(self.dependencies, (list, tuple)) or any(
            not isinstance(dependency, NeedDependency)
            for dependency in self.dependencies
        ):
            raise ValueError("dependencies must contain NeedDependency values")
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        if not isinstance(self.model_called, bool):
            raise ValueError("model_called must be a bool")
        if self.status == "ready" and not self.needs:
            raise ValueError("ready understanding must contain at least one need")

    @classmethod
    def ready(
        cls,
        needs: Sequence[UserNeed],
        *,
        dependencies: Sequence[NeedDependency] = (),
        raw_response: object | None = None,
        model_called: bool = False,
    ) -> "UnderstandingResult":
        return cls(
            "ready",
            tuple(needs),
            dependencies=tuple(dependencies),
            raw_response=raw_response,
            model_called=model_called,
        )

    @classmethod
    def clarify(
        cls,
        question: str,
        *,
        raw_response: object | None = None,
        model_called: bool = False,
    ) -> "UnderstandingResult":
        return cls(
            "clarify",
            (),
            clarification_question=question,
            raw_response=raw_response,
            model_called=model_called,
        )

    @classmethod
    def error(
        cls,
        reason: str,
        *,
        raw_response: object | None = None,
        model_called: bool = False,
    ) -> "UnderstandingResult":
        return cls(
            "error",
            (),
            error_reason=reason,
            raw_response=raw_response,
            model_called=model_called,
        )
