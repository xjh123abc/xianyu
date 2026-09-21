"""Small, transport-free contracts shared by Xianyu experts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, get_args


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
QueryTarget = Literal[
    "availability.sale_status",
    "history.repair_history",
    "history.disassembly_history",
    "history.drop_history",
    "function.shutter",
    "function.overall",
    "condition.summary",
    "condition.scratches",
    "condition.dents",
    "condition.known_issues",
    "lens.details",
    "lens.focal_length_mm",
    "accessories.items",
    "accessories.completeness",
    "accessories.original",
    "accessories.manual_or_packaging",
    "identity.model",
    "product_info.production_year",
    "product_info.beginner_suitability",
    "product_info.usage",
    "product_info.sale_reason",
    "price.listed_price",
    "price.minimum",
    "price.offer",
    "price.additional_discount",
    "shipping.dispatch_time",
    "shipping.ship_from",
    "shipping.carrier",
    "shipping.fee",
    "after_sale.return_policy",
    "after_sale.transaction_channel",
    "after_sale.description_policy",
    "after_sale.inspection_confirmation",
    "seller_rule.general",
    "product.model_knowledge",
    "greeting",
]
VALID_QUERY_TARGETS = frozenset(get_args(QueryTarget))


@dataclass(frozen=True, slots=True)
class ExpertTask:
    """One planner-owned question for exactly one Xianyu domain expert.

    ``question_fragment`` is retained as a wire-compatible name, but it is
    now always a complete buyer sub-question rather than a keyword slice.
    ``query_target`` is the planner's explicit retrieval contract; experts
    must use it instead of classifying the question again.
    """

    task_id: str
    expert: ExpertName
    question_fragment: str
    normalized_question: str
    knowledge_scope: ExpertKnowledgeScope
    query_target: QueryTarget
    transaction_conditions: Mapping[str, object] = field(default_factory=dict)
    depends_on_task_ids: tuple[str, ...] = ()
    original_question: str = ""

    def __post_init__(self) -> None:
        """Expose the complete source question in both old and new contracts."""

        original = self.original_question.strip() or self.question_fragment
        object.__setattr__(self, "original_question", original)
        object.__setattr__(self, "question_fragment", original)


@dataclass(frozen=True, slots=True)
class ExpertContext:
    """Read-only turn context shared by the future expert orchestrator."""

    query: str
    item: Mapping[str, object] | None
    history: Sequence[Mapping[str, object]] = ()
    xianyu_context: Mapping[str, object] = field(default_factory=dict)
    # Owned by the orchestrator and never persisted as buyer/session state.
    deadline: float | None = None
    # Legacy direct callers retain their historical wording veto.  The unified
    # TaskExecutor path keeps evidence checks but does not apply that veto.
    use_legacy_text_guard: bool = True


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
