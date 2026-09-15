"""Small, transport-free contracts shared by Xianyu experts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PriceDecisionStatus = Literal["answered", "handoff"]
ShippingCondition = Literal["seller_pays", "buyer_pays"]


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
