"""Deterministic pricing expert for one confirmed Xianyu item."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import re

from app.services.intent_router import IntentMatch
from app.services.xianyu.experts.contracts import PriceDecision, ShippingCondition


@dataclass(frozen=True, slots=True)
class _PricePolicy:
    original_price_cents: int
    minor_discount_cents: int
    buyer_pays_shipping_discount_cents: int | None
    buyer_pays_shipping_stacks_minor_discount: bool


class PriceAgent:
    """Compute authorised prices from one item's validated seller facts only."""

    _BUYER_PAYS_SHIPPING_TERMS = (
        "不用包邮",
        "不包邮",
        "出邮费",
        "出运费",
        "我出邮费",
        "承担运费",
        "自付运费",
    )
    _UNAUTHORISED_CONDITION_TERMS = (
        "自提",
        "自取",
        "面交",
        "只买机身",
        "不要镜头",
        "拆卖",
    )
    _ADDITIONAL_DISCOUNT_TERMS = (
        "再便宜",
        "再优惠",
        "再少",
        "再刀",
        "还能少",
    )
    _MINOR_DISCOUNT_POLICY = re.compile(
        r"^\s*累计最多小刀\s*(\d{1,7}(?:\.\d{1,2})?)\s*元\s*$"
    )
    _BUYER_PAYS_POLICY = re.compile(
        r"^\s*不包邮商品价减\s*(\d{1,7}(?:\.\d{1,2})?)\s*元\s*，\s*可叠加小刀\s*$"
    )
    _EXPLICIT_OFFER_PATTERNS = (
        re.compile(
            r"(?:我\s*(?:出|给)|(?:出|报)价|按|到手价)\s*"
            r"[¥￥]?\s*(\d{1,7}(?:\.\d{1,2})?)\s*(?:元|块|rmb)?",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?<!\d)[¥￥]?\s*(\d{1,7}(?:\.\d{1,2})?)\s*(?:元|块|rmb)?"
            r"\s*(?:(?:包邮|不包邮|自提|自取|面交|到手)?\s*)"
            r"(?:可以|行吗|能出|能收|卖吗|吗)",
            re.IGNORECASE,
        ),
    )

    def decide(
        self,
        query: str,
        item: Mapping[str, object],
        match: IntentMatch,
    ) -> PriceDecision:
        """Return an answer or handoff reason without performing any I/O."""

        facts = self._facts(item)
        conflict = self._relevant_conflict(facts)
        if conflict is not None:
            return PriceDecision.handoff(conflict)

        sale_status = item.get("sale_status")
        if sale_status == "sold":
            return PriceDecision.answered("这件已经出掉了。")
        if sale_status != "listed":
            return PriceDecision.handoff("sale_status_unavailable")

        original_price_cents = item.get("listed_price_cents")
        if (
            not isinstance(original_price_cents, int)
            or isinstance(original_price_cents, bool)
            or original_price_cents < 0
        ):
            return PriceDecision.handoff("listed_price_unavailable")

        if match.intent == "PRICE":
            return PriceDecision.answered(
                f"这件标价是 {self._format_cents(original_price_cents)}。",
                original_price_cents=original_price_cents,
            )
        if match.intent != "BARGAIN":
            return PriceDecision.handoff("price_intent_unavailable")

        lowered = query.casefold()
        if any(term in lowered for term in self._UNAUTHORISED_CONDITION_TERMS):
            return PriceDecision.handoff("unsupported_price_condition")
        if any(term in lowered for term in self._ADDITIONAL_DISCOUNT_TERMS):
            return PriceDecision.handoff("additional_discount_not_authorised")

        buyer_pays_shipping = any(
            term in lowered for term in self._BUYER_PAYS_SHIPPING_TERMS
        )
        includes_shipping = bool(re.search(r"(?<![不用])包邮", lowered))
        asks_both_shipping_options = buyer_pays_shipping and (
            includes_shipping or lowered.count("最低") >= 2
        )
        buyer_offer_cents = self._buyer_offer_cents(query)
        if asks_both_shipping_options and buyer_offer_cents is not None:
            return PriceDecision.handoff("ambiguous_shipping_condition_for_offer")

        policy_or_reason = self._parse_policy(
            item,
            facts,
            needs_buyer_pays_shipping=buyer_pays_shipping,
        )
        if isinstance(policy_or_reason, str):
            return PriceDecision.handoff(policy_or_reason)
        policy = policy_or_reason

        if asks_both_shipping_options:
            seller_pays_minimum = self._minimum_price(policy, "seller_pays")
            buyer_pays_minimum = self._minimum_price(policy, "buyer_pays")
            if seller_pays_minimum is None or buyer_pays_minimum is None:
                return PriceDecision.handoff("buyer_pays_shipping_policy_unavailable")
            return PriceDecision.answered(
                "包邮最低 "
                f"{self._format_cents(seller_pays_minimum)}；"
                f"不包邮的话最低 {self._format_cents(buyer_pays_minimum)}。",
                original_price_cents=policy.original_price_cents,
                minimum_price_cents=buyer_pays_minimum,
            )

        shipping_condition: ShippingCondition = (
            "buyer_pays" if buyer_pays_shipping else "seller_pays"
        )
        minimum_price_cents = self._minimum_price(policy, shipping_condition)
        if minimum_price_cents is None:
            return PriceDecision.handoff("buyer_pays_shipping_policy_unavailable")
        effective_price_cents = self._effective_price(policy, shipping_condition)

        if buyer_offer_cents is not None:
            if buyer_offer_cents < minimum_price_cents:
                return PriceDecision.handoff(
                    "buyer_offer_below_authorised_minimum:"
                    f"offer={self._format_cents(buyer_offer_cents)};"
                    f"minimum={self._format_cents(minimum_price_cents)}",
                    original_price_cents=policy.original_price_cents,
                    effective_price_cents=effective_price_cents,
                    minimum_price_cents=minimum_price_cents,
                    shipping_condition=shipping_condition,
                    buyer_offer_cents=buyer_offer_cents,
                )
            return PriceDecision.answered(
                f"可以，{self._format_cents(buyer_offer_cents)} 可以拍。",
                original_price_cents=policy.original_price_cents,
                effective_price_cents=effective_price_cents,
                minimum_price_cents=minimum_price_cents,
                shipping_condition=shipping_condition,
                buyer_offer_cents=buyer_offer_cents,
            )

        if shipping_condition == "buyer_pays":
            answer = f"不包邮的话最低 {self._format_cents(minimum_price_cents)} 可以拍。"
        elif policy.minor_discount_cents:
            answer = f"最低 {self._format_cents(minimum_price_cents)} 可以拍。"
        else:
            answer = f"标价 {self._format_cents(minimum_price_cents)}，这个价格可以拍。"
        return PriceDecision.answered(
            answer,
            original_price_cents=policy.original_price_cents,
            effective_price_cents=effective_price_cents,
            minimum_price_cents=minimum_price_cents,
            shipping_condition=shipping_condition,
        )

    def _parse_policy(
        self,
        item: Mapping[str, object],
        facts: Mapping[str, object],
        *,
        needs_buyer_pays_shipping: bool,
    ) -> _PricePolicy | str:
        original_price_cents = item["listed_price_cents"]
        assert isinstance(original_price_cents, int)

        negotiation = self._known_text(facts.get("negotiation"))
        if negotiation == "firm":
            minor_discount_cents = 0
        elif negotiation == "negotiable":
            shipping = facts.get("shipping")
            policy_text = (
                self._known_text(shipping.get("negotiation_policy"))
                if isinstance(shipping, Mapping)
                else None
            )
            minor_discount_cents = self._parse_discount(
                policy_text,
                self._MINOR_DISCOUNT_POLICY,
            )
            if minor_discount_cents is None:
                return "minor_discount_policy_unavailable"
        else:
            return "negotiation_policy_unavailable"

        if minor_discount_cents > original_price_cents:
            return "minor_discount_exceeds_listed_price"

        buyer_pays_discount_cents: int | None = None
        stacks_minor_discount = False
        if needs_buyer_pays_shipping:
            shipping = facts.get("shipping")
            policy_text = (
                self._known_text(shipping.get("negotiation_express_policy"))
                if isinstance(shipping, Mapping)
                else None
            )
            buyer_pays_discount_cents = self._parse_discount(
                policy_text,
                self._BUYER_PAYS_POLICY,
            )
            if buyer_pays_discount_cents is None:
                return "buyer_pays_shipping_policy_unavailable"
            stacks_minor_discount = True
            if buyer_pays_discount_cents > original_price_cents:
                return "buyer_pays_shipping_discount_exceeds_listed_price"

        return _PricePolicy(
            original_price_cents=original_price_cents,
            minor_discount_cents=minor_discount_cents,
            buyer_pays_shipping_discount_cents=buyer_pays_discount_cents,
            buyer_pays_shipping_stacks_minor_discount=stacks_minor_discount,
        )

    @staticmethod
    def _facts(item: Mapping[str, object]) -> Mapping[str, object]:
        facts = item.get("facts")
        return facts if isinstance(facts, Mapping) else {}

    @staticmethod
    def _known_text(value: object) -> str | None:
        return value.strip() if isinstance(value, str) and value.strip() != "unknown" else None

    @staticmethod
    def _relevant_conflict(facts: Mapping[str, object]) -> str | None:
        conflicts = facts.get("fact_conflicts")
        if not isinstance(conflicts, list):
            return None
        relevant = {"negotiation", "shipping", "sale_status", "listed_price_cents"}
        conflict = next(
            (entry for entry in conflicts if isinstance(entry, str) and entry in relevant),
            None,
        )
        return f"price_fact_conflict:{conflict}" if conflict is not None else None

    @classmethod
    def _parse_discount(cls, value: str | None, pattern: re.Pattern[str]) -> int | None:
        if value is None:
            return None
        match = pattern.fullmatch(value)
        if match is None:
            return None
        try:
            cents = (Decimal(match.group(1)) * 100).quantize(
                Decimal("1"), rounding=ROUND_HALF_UP
            )
        except (InvalidOperation, ValueError):
            return None
        return int(cents) if cents >= 0 else None

    @classmethod
    def _buyer_offer_cents(cls, query: str) -> int | None:
        for pattern in cls._EXPLICIT_OFFER_PATTERNS:
            match = pattern.search(query)
            if match is None:
                continue
            prefix = query[max(0, match.start(1) - 8) : match.start(1)]
            if any(term in prefix for term in ("便宜", "优惠", "少", "减")):
                continue
            try:
                cents = (Decimal(match.group(1)) * 100).quantize(
                    Decimal("1"), rounding=ROUND_HALF_UP
                )
            except (InvalidOperation, ValueError):
                continue
            if cents >= 0:
                return int(cents)
        return None

    @staticmethod
    def _effective_price(
        policy: _PricePolicy,
        shipping_condition: ShippingCondition,
    ) -> int:
        if shipping_condition == "seller_pays":
            return policy.original_price_cents
        assert policy.buyer_pays_shipping_discount_cents is not None
        return policy.original_price_cents - policy.buyer_pays_shipping_discount_cents

    @classmethod
    def _minimum_price(
        cls,
        policy: _PricePolicy,
        shipping_condition: ShippingCondition,
    ) -> int | None:
        if shipping_condition == "buyer_pays" and not policy.buyer_pays_shipping_stacks_minor_discount:
            return None
        effective_price = cls._effective_price(policy, shipping_condition)
        minimum_price = effective_price - policy.minor_discount_cents
        return minimum_price if minimum_price >= 0 else None

    @staticmethod
    def _format_cents(cents: int) -> str:
        return f"¥{cents / 100:.2f}"
