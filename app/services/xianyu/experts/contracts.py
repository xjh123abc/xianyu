"""Small, transport-free contracts shared by Xianyu experts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal


PriceDecisionStatus = Literal["answered", "handoff"]
ShippingCondition = Literal["seller_pays", "buyer_pays"]
ExpertName = Literal["product", "price", "service"]
ExpertKnowledgeScope = Literal[
    "item_fact",
    "model_knowledge",
    "seller_rule",
    "greeting",
]
ExpertResultStatus = Literal["answered", "handoff"]


@dataclass(frozen=True, slots=True)
class ExpertTask:
    """One planner-owned question for exactly one Xianyu domain expert."""

    task_id: str
    expert: ExpertName
    question_fragment: str
    normalized_question: str
    knowledge_scope: ExpertKnowledgeScope
    transaction_conditions: Mapping[str, object] = field(default_factory=dict)
    depends_on_task_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExpertContext:
    """Read-only turn context shared by the future expert orchestrator."""

    query: str
    item: Mapping[str, object] | None
    history: Sequence[Mapping[str, object]] = ()
    xianyu_context: Mapping[str, object] = field(default_factory=dict)
    # Owned by the orchestrator and never persisted as buyer/session state.
    deadline: float | None = None


@dataclass(frozen=True, slots=True)
class ExpertResult:
    """One evidence-bound expert outcome; experts never perform delivery."""

    task_id: str
    expert: ExpertName
    status: ExpertResultStatus
    answer: str | None = None
    sources: tuple[Mapping[str, object], ...] = ()
    missing_fields: tuple[str, ...] = ()
    reason: str | None = None

    @classmethod
    def answered(
        cls,
        task: ExpertTask,
        answer: str,
        *,
        sources: Sequence[Mapping[str, object]] = (),
    ) -> "ExpertResult":
        return cls(
            task_id=task.task_id,
            expert=task.expert,
            status="answered",
            answer=answer,
            sources=tuple(sources),
        )

    @classmethod
    def handoff(
        cls,
        task: ExpertTask,
        reason: str,
        *,
        missing_fields: Sequence[str] = (),
        sources: Sequence[Mapping[str, object]] = (),
    ) -> "ExpertResult":
        return cls(
            task_id=task.task_id,
            expert=task.expert,
            status="handoff",
            sources=tuple(sources),
            missing_fields=tuple(missing_fields),
            reason=reason,
        )


@dataclass(frozen=True, slots=True)
class PriceDecision:
    """One deterministic price-expert result, independent of channel delivery."""

    status: PriceDecisionStatus
    answer: str | None = None
    reason: str | None = None
    original_price_cents: int | None = None
    effective_price_cents: int | None = None
    minimum_price_cents: int | None = None
    shipping_condition: ShippingCondition | None = None
    buyer_offer_cents: int | None = None

    @classmethod
    def answered(
        cls,
        answer: str,
        *,
        original_price_cents: int | None = None,
        effective_price_cents: int | None = None,
        minimum_price_cents: int | None = None,
        shipping_condition: ShippingCondition | None = None,
        buyer_offer_cents: int | None = None,
    ) -> "PriceDecision":
        return cls(
            "answered",
            answer=answer,
            original_price_cents=original_price_cents,
            effective_price_cents=effective_price_cents,
            minimum_price_cents=minimum_price_cents,
            shipping_condition=shipping_condition,
            buyer_offer_cents=buyer_offer_cents,
        )

    @classmethod
    def handoff(
        cls,
        reason: str,
        *,
        original_price_cents: int | None = None,
        effective_price_cents: int | None = None,
        minimum_price_cents: int | None = None,
        shipping_condition: ShippingCondition | None = None,
        buyer_offer_cents: int | None = None,
    ) -> "PriceDecision":
        return cls(
            "handoff",
            reason=reason,
            original_price_cents=original_price_cents,
            effective_price_cents=effective_price_cents,
            minimum_price_cents=minimum_price_cents,
            shipping_condition=shipping_condition,
            buyer_offer_cents=buyer_offer_cents,
        )
