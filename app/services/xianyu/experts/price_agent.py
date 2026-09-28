"""Deterministic pricing expert for one confirmed Xianyu item."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import re
from typing import Literal
from uuid import uuid4

from app.services.intent_router import IntentMatch
from app.services.xianyu.experts.contracts import (
    ExpertContext,
    ExpertResult,
    ExpertTask,
    PriceDecision,
    ShippingCondition,
)
from app.services.xianyu.responses import item_source
from app.services.negotiation_policy import (
    NegotiationPolicy,
    NegotiationPolicyError,
    NegotiationPolicyStore,
)


@dataclass(frozen=True, slots=True)
class _PricePolicy:
    original_price_cents: int
    minor_discount_cents: int
    buyer_pays_shipping_discount_cents: int | None
    buyer_pays_shipping_stacks_minor_discount: bool


_NegotiationAction = Literal["HOLD", "COUNTER", "ACCEPT", "REJECT", "CLARIFY"]


@dataclass(frozen=True, slots=True)
class _NegotiationReplyPlan:
    decision: _NegotiationAction
    shipping_condition: ShippingCondition | None = None
    listed_price_cents: int | None = None
    current_offer_cents: int | None = None
    counter_offer_cents: int | None = None
    accepted_offer_cents: int | None = None
    buyer_offer_cents: int | None = None
    selling_point: str | None = None
    message: str | None = None


class PriceAgent:
    """Compute authorised prices from one item's validated seller facts only."""

    def __init__(self, policy_store: NegotiationPolicyStore | None = None) -> None:
        self.policy_store = policy_store or NegotiationPolicyStore()

    async def run(
        self,
        tasks: list[ExpertTask],
        context: ExpertContext,
    ) -> list[ExpertResult]:
        """Handle a batch of price tasks through the shared expert contract."""

        results: list[ExpertResult] = []
        for task in tasks:
            if task.expert != "price":
                results.append(ExpertResult.handoff(task, "task_expert_mismatch"))
                continue
            if context.item is None:
                results.append(
                    ExpertResult.handoff(
                        task,
                        "item_context_unavailable",
                        missing_fields=("item",),
                    )
                )
                continue

            request_kind = {
                "price.listed_price": "listed_price",
                "price.minimum": "minimum",
                "price.offer": "offer",
                "price.additional_discount": "additional_discount",
                "price.confirm": "confirm",
            }.get(
                task.query_target,
                str(task.transaction_conditions.get("request_kind", "listed_price")),
            )
            match = IntentMatch(
                "PRICE" if request_kind == "listed_price" else "BARGAIN",
                ("listed_price_cents",)
                if request_kind == "listed_price"
                else ("listed_price_cents", "negotiation", "shipping"),
                "rule",
            )
            # Keep the task's own query contract intact while allowing a
            # deterministic price comparison to see both clauses of the one
            # buyer message (for example, package shipping vs. self-paid).
            original_query = context.xianyu_context.get("original_query")
            decision_query = original_query if isinstance(original_query, str) else context.query
            decision = self.decide(
                decision_query,
                context.item,
                match,
                negotiation=context.negotiation,
                turn_id=context.turn_id,
                request_kind=request_kind,
                shipping_comparison=(
                    task.transaction_conditions.get("shipping_comparison") is True
                ),
            )
            sources = (item_source(context.item),)
            if decision.status == "answered" and decision.answer:
                results.append(
                    ExpertResult.answered(
                        task,
                        decision.answer,
                        sources=sources,
                        state_proposal=decision.state_proposal,
                    )
                )
            else:
                results.append(
                    ExpertResult.handoff(
                        task,
                        decision.reason or "price_decision_unavailable",
                        sources=sources,
                    )
                )
        return results

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
        "便宜点",
        "便宜一点",
        "优惠点",
        "优惠一点",
        "少点",
        "少一点",
        "刀一下",
        "再便宜",
        "再优惠",
        "再少",
        "再刀",
        "还能少",
        "还能便宜",
        "再让",
        "让一点",
        "最后再",
        "马上买",
        "现在拍",
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
        *,
        negotiation: Mapping[str, object] | None = None,
        turn_id: str | None = None,
        request_kind: str | None = None,
        shipping_comparison: bool = False,
    ) -> PriceDecision:
        """Return a policy-backed decision, falling back to legacy one-turn rules.

        Only explicitly configured private policies receive the continuous
        bargaining behaviour.  Existing seller facts remain a strict one-turn
        compatibility path and are never expanded into invented price tiers.
        """

        # A known listing price answers a price question even when the platform
        # cannot currently confirm whether the listing is still active.  Keep
        # the stricter sale-status and policy gates for offers and discounts.
        if match.intent == "PRICE":
            return self._listed_price_decision(item)

        try:
            policy = self.policy_store.get(item)
        except NegotiationPolicyError as exc:
            return PriceDecision.handoff(str(exc))
        if policy is None or match.intent != "BARGAIN":
            return self._legacy_decide(query, item, match)
        return self._policy_decide(
            query,
            item,
            policy,
            negotiation or {},
            turn_id,
            request_kind,
            shipping_comparison,
        )

    def _listed_price_decision(self, item: Mapping[str, object]) -> PriceDecision:
        """Answer from the current public listing price without negotiation data."""

        if item.get("sale_status") == "sold":
            return PriceDecision.answered("这件已经出掉了。")

        price_cents = item.get("listed_price_cents")
        if (
            not isinstance(price_cents, int)
            or isinstance(price_cents, bool)
            or price_cents < 0
        ):
            return PriceDecision.handoff("listed_price_unavailable")

        facts = self._facts(item)
        conflicts = facts.get("fact_conflicts")
        if (
            isinstance(conflicts, list)
            and "listed_price_cents" in conflicts
            and item.get("data_source") != "seller_snapshot+xianyu_platform"
        ):
            return PriceDecision.handoff("price_fact_conflict:listed_price_cents")

        return PriceDecision.answered(
            f"这件标价是 {self._format_cents(price_cents)}。",
            original_price_cents=price_cents,
        )

    def _policy_decide(
        self,
        query: str,
        item: Mapping[str, object],
        policy: NegotiationPolicy,
        negotiation: Mapping[str, object],
        turn_id: str | None,
        request_kind: str | None,
        shipping_comparison: bool,
    ) -> PriceDecision:
        """Apply one private tier policy without exposing its floor or tiers."""

        facts = self._facts(item)
        conflict = self._relevant_conflict(facts)
        if conflict is not None:
            return PriceDecision.handoff(conflict)
        if item.get("sale_status") == "sold":
            return PriceDecision.answered("这件已经出掉了。")
        if item.get("sale_status") != "listed":
            return PriceDecision.handoff("sale_status_unavailable")

        lowered = query.casefold()
        if any(term in lowered for term in self._UNAUTHORISED_CONDITION_TERMS):
            return PriceDecision.handoff("unsupported_price_condition")

        buyer_pays_shipping = any(
            term in lowered for term in self._BUYER_PAYS_SHIPPING_TERMS
        )
        includes_shipping = bool(re.search(r"(?<![不用])包邮", lowered))
        asks_both_shipping_options = shipping_comparison or (buyer_pays_shipping and (
            includes_shipping or lowered.count("最低") >= 2
        ))
        buyer_offer_cents = self._buyer_offer_cents(query)
        if asks_both_shipping_options:
            if buyer_offer_cents is not None:
                return PriceDecision.answered(
                    self._reply_from_plan(
                        _NegotiationReplyPlan(
                            "CLARIFY",
                            message="你这个报价是想包邮还是不包邮？运费条件确认下我再给你准话。",
                        )
                    ),
                    original_price_cents=policy.listed_price_cents,
                    buyer_offer_cents=buyer_offer_cents,
                )
            seller_discount = self._discount_amount_phrase(
                policy.listed_price_cents,
                policy.seller_pays.offers_cents[0],
            )
            buyer_discount = self._discount_amount_phrase(
                policy.listed_price_cents,
                policy.buyer_pays.offers_cents[0],
            )
            return PriceDecision.answered(
                f"包邮可以{seller_discount}；不包邮可以{buyer_discount}。"
                f"{self._selling_point_sentence(self._selling_point(item))}",
                original_price_cents=policy.listed_price_cents,
            )

        active = self._active_state(negotiation, policy)
        stored_shipping = active.get("shipping_condition")
        shipping_condition: ShippingCondition = (
            "buyer_pays"
            if buyer_pays_shipping
            else stored_shipping
            if stored_shipping in {"seller_pays", "buyer_pays"}
            else "seller_pays"
        )
        shipping_policy = policy.for_shipping(shipping_condition)
        current_round = int(active.get("round", 0))
        state_shipping = active.get("shipping_condition")
        switched_shipping = state_shipping in {"seller_pays", "buyer_pays"} and state_shipping != shipping_condition
        current_offer = None if switched_shipping else active.get("last_ai_offer")
        pending = active.get("pending_offer")

        if active.get("offer_status") == "unknown":
            pending_price = pending.get("price_cents") if isinstance(pending, Mapping) else None
            price = pending_price if _is_cents(pending_price) else current_offer
            if _is_cents(price):
                return PriceDecision.answered(
                    "上一轮报价 "
                    f"{self._format_cents(price)}{self._shipping_label(shipping_condition)}"
                    "的发送状态还待确认，暂时不能继续调整价格。",
                    original_price_cents=policy.listed_price_cents,
                    effective_price_cents=price,
                    shipping_condition=shipping_condition,
                )
            return PriceDecision.handoff("negotiation_delivery_unknown")

        # A retried platform message must return its existing draft, rather
        # than minting another proposal or consuming another tier.
        if (
            isinstance(pending, Mapping)
            and active.get("offer_status") == "generated"
            and isinstance(turn_id, str)
            and pending.get("turn_id") == turn_id
            and pending.get("shipping_condition") == shipping_condition
            and _is_cents(pending.get("price_cents"))
        ):
            pending_price = pending["price_cents"]
            last_buyer_offer = active.get("last_buyer_offer")
            if _is_cents(last_buyer_offer) and last_buyer_offer < pending_price:
                answer = self._reply_from_plan(
                    _NegotiationReplyPlan(
                        "REJECT",
                        shipping_condition=shipping_condition,
                        listed_price_cents=policy.listed_price_cents,
                        current_offer_cents=pending_price,
                        counter_offer_cents=pending_price,
                        buyer_offer_cents=last_buyer_offer,
                        selling_point=self._selling_point(item),
                    )
                )
            else:
                answer = self._reply_from_plan(
                    _NegotiationReplyPlan(
                        "COUNTER",
                        shipping_condition=shipping_condition,
                        listed_price_cents=policy.listed_price_cents,
                        counter_offer_cents=pending_price,
                        selling_point=self._selling_point(item),
                    )
                )
            return PriceDecision.answered(
                answer,
                original_price_cents=policy.listed_price_cents,
                effective_price_cents=pending_price,
                shipping_condition=shipping_condition,
                state_proposal={
                    "offer_status": "generated",
                    "pending_offer": dict(pending),
                },
            )

        if self._accepts_previous_offer(lowered, request_kind) and _is_cents(current_offer):
            return PriceDecision.answered(
                self._reply_from_plan(
                    _NegotiationReplyPlan(
                        "ACCEPT",
                        shipping_condition=shipping_condition,
                        listed_price_cents=policy.listed_price_cents,
                        accepted_offer_cents=current_offer,
                        selling_point=self._selling_point(item),
                        message=(
                            "可以，就按 "
                            f"{self._format_cents(current_offer)}"
                            f"{self._shipping_label(shipping_condition)}，"
                            "目前仅确认报价，尚未实际改价或创建订单。"
                        ),
                    )
                ),
                original_price_cents=policy.listed_price_cents,
                effective_price_cents=current_offer,
                shipping_condition=shipping_condition,
                state_proposal={
                    "item_id": policy.item_id,
                    "policy_version": policy.version,
                    "round": current_round,
                    "last_ai_offer": current_offer,
                    "shipping_condition": shipping_condition,
                    "offer_status": "confirmed",
                    "pending_offer": None,
                },
            )

        if buyer_offer_cents is not None:
            visible_offer = (
                current_offer
                if _is_cents(current_offer)
                else shipping_policy.offers_cents[
                    min(max(current_round - 1, 0), len(shipping_policy.offers_cents) - 1)
                ]
            )
            if buyer_offer_cents < visible_offer:
                proposal = (
                    None
                    if _is_cents(current_offer)
                    else self._proposal(
                        policy,
                        turn_id,
                        visible_offer,
                        shipping_condition,
                        min(current_round + 1, len(shipping_policy.offers_cents)),
                        buyer_offer_cents,
                    )
                )
                return PriceDecision.answered(
                    self._reply_from_plan(
                        _NegotiationReplyPlan(
                            "REJECT",
                            shipping_condition=shipping_condition,
                            listed_price_cents=policy.listed_price_cents,
                            current_offer_cents=visible_offer,
                            counter_offer_cents=visible_offer,
                            buyer_offer_cents=buyer_offer_cents,
                            selling_point=self._selling_point(item),
                        )
                    ),
                    original_price_cents=policy.listed_price_cents,
                    effective_price_cents=visible_offer,
                    minimum_price_cents=shipping_policy.floor_cents,
                    shipping_condition=shipping_condition,
                    buyer_offer_cents=buyer_offer_cents,
                    state_proposal=proposal,
                )
            if buyer_offer_cents <= policy.listed_price_cents:
                proposal = self._proposal(
                    policy,
                    turn_id,
                    buyer_offer_cents,
                    shipping_condition,
                    max(1, min(current_round, len(shipping_policy.offers_cents))),
                    buyer_offer_cents,
                )
                return PriceDecision.answered(
                    self._reply_from_plan(
                        _NegotiationReplyPlan(
                            "ACCEPT",
                            shipping_condition=shipping_condition,
                            listed_price_cents=policy.listed_price_cents,
                            accepted_offer_cents=buyer_offer_cents,
                            buyer_offer_cents=buyer_offer_cents,
                            selling_point=self._selling_point(item),
                        )
                    ),
                    original_price_cents=policy.listed_price_cents,
                    effective_price_cents=buyer_offer_cents,
                    minimum_price_cents=shipping_policy.floor_cents,
                    shipping_condition=shipping_condition,
                    buyer_offer_cents=buyer_offer_cents,
                    state_proposal=proposal,
                )

        asks_additional_discount = any(
            term in lowered for term in self._ADDITIONAL_DISCOUNT_TERMS
        )
        asks_generic_discount = asks_additional_discount and "最低" not in lowered
        if not _is_cents(current_offer) and asks_generic_discount and not buyer_pays_shipping:
            return PriceDecision.answered(
                self._reply_from_plan(
                    _NegotiationReplyPlan(
                        "CLARIFY",
                        selling_point=self._selling_point(item),
                        message=(
                            "你想多少收？"
                            f"{self._selling_point_sentence(self._selling_point(item))}"
                        ),
                    )
                ),
                original_price_cents=policy.listed_price_cents,
                shipping_condition=shipping_condition,
            )
        # Once a valid offer exists, generic pressure such as "再便宜点" or
        # "马上买" holds the current price.  Only an explicit buyer offer can
        # change the price from here.
        if _is_cents(current_offer) and (
            asks_additional_discount or request_kind in {"minimum", "additional_discount"}
        ):
            return PriceDecision.answered(
                self._reply_from_plan(
                    _NegotiationReplyPlan(
                        "HOLD",
                        shipping_condition=shipping_condition,
                        listed_price_cents=policy.listed_price_cents,
                        current_offer_cents=current_offer,
                        selling_point=self._selling_point(item),
                    )
                ),
                original_price_cents=policy.listed_price_cents,
                effective_price_cents=current_offer,
                minimum_price_cents=shipping_policy.floor_cents,
                shipping_condition=shipping_condition,
            )

        # At the final authorised tier, repeat the current price indefinitely;
        # no wording can make an unbounded additional discount appear.
        if current_round >= len(shipping_policy.offers_cents) and _is_cents(current_offer):
            return PriceDecision.answered(
                self._reply_from_plan(
                    _NegotiationReplyPlan(
                        "HOLD",
                        shipping_condition=shipping_condition,
                        listed_price_cents=policy.listed_price_cents,
                        current_offer_cents=current_offer,
                        selling_point=self._selling_point(item),
                    )
                ),
                original_price_cents=policy.listed_price_cents,
                effective_price_cents=current_offer,
                minimum_price_cents=shipping_policy.floor_cents,
                shipping_condition=shipping_condition,
            )

        next_round = min(
            current_round + 1
            if not switched_shipping or asks_additional_discount
            else max(current_round, 1),
            len(shipping_policy.offers_cents),
        )
        offer = shipping_policy.offers_cents[next_round - 1]
        proposal = self._proposal(
            policy,
            turn_id,
            offer,
            shipping_condition,
            next_round,
            buyer_offer_cents,
        )
        return PriceDecision.answered(
            self._reply_from_plan(
                _NegotiationReplyPlan(
                    "COUNTER",
                    shipping_condition=shipping_condition,
                    listed_price_cents=policy.listed_price_cents,
                    counter_offer_cents=offer,
                    selling_point=self._selling_point(item),
                )
            ),
            original_price_cents=policy.listed_price_cents,
            effective_price_cents=offer,
            minimum_price_cents=shipping_policy.floor_cents,
            shipping_condition=shipping_condition,
            state_proposal=proposal,
        )

    @staticmethod
    def _active_state(
        negotiation: Mapping[str, object], policy: NegotiationPolicy
    ) -> Mapping[str, object]:
        item_id = negotiation.get("item_id")
        version = negotiation.get("policy_version")
        if (
            not isinstance(item_id, str)
            or item_id.strip().upper() != policy.item_id
            or version != policy.version
        ):
            return {}
        return negotiation

    @staticmethod
    def _accepts_previous_offer(lowered: str, request_kind: str | None) -> bool:
        if request_kind == "offer":
            return False
        return any(term in lowered for term in ("就按", "刚才那个价", "刚才说的", "这个价"))

    def _proposal(
        self,
        policy: NegotiationPolicy,
        turn_id: str | None,
        price_cents: int,
        shipping_condition: ShippingCondition,
        round_number: int,
        buyer_offer_cents: int | None,
    ) -> dict[str, object]:
        stable_turn_id = turn_id.strip() if isinstance(turn_id, str) and turn_id.strip() else f"direct:{uuid4()}"
        pending_offer = {
            "proposal_id": str(uuid4()),
            "turn_id": stable_turn_id,
            "item_id": policy.item_id,
            "policy_version": policy.version,
            "price_cents": price_cents,
            "shipping_condition": shipping_condition,
            "round": round_number,
        }
        proposal: dict[str, object] = {
            "item_id": policy.item_id,
            "policy_version": policy.version,
            "offer_status": "generated",
            "pending_offer": pending_offer,
        }
        if buyer_offer_cents is not None:
            proposal["last_buyer_offer"] = buyer_offer_cents
        return proposal

    @staticmethod
    def _shipping_label(condition: ShippingCondition) -> str:
        return "包邮" if condition == "seller_pays" else "（不包邮）"

    def _reply_from_plan(self, plan: _NegotiationReplyPlan) -> str:
        """Render the authorised price decision without changing any amount."""

        if plan.message is not None:
            return plan.message
        if plan.decision == "CLARIFY":
            return "你想确认的是包邮价还是不包邮价？我按条件给你准价。"
        if plan.decision == "ACCEPT":
            assert _is_cents(plan.accepted_offer_cents)
            return (
                "可以，"
                f"{self._format_cents(plan.accepted_offer_cents)}"
                f"{self._shipping_label(plan.shipping_condition or 'seller_pays')}"
                "可以拍。"
            )
        if plan.decision == "HOLD":
            assert _is_cents(plan.current_offer_cents)
            discount = self._discount_phrase(
                plan.listed_price_cents,
                plan.current_offer_cents,
                plan.shipping_condition or "seller_pays",
            )
            if discount:
                return (
                    f"当前已经{discount}，这次就不再往下调了。"
                    f"{self._selling_point_sentence(plan.selling_point)}"
                )
            return (
                "当前这个价这次不再往下调了。"
                f"{self._selling_point_sentence(plan.selling_point)}"
            )
        if plan.decision == "REJECT":
            assert _is_cents(plan.current_offer_cents)
            buyer = (
                f"{self._format_cents(plan.buyer_offer_cents)} "
                if _is_cents(plan.buyer_offer_cents)
                else "这个价格"
            )
            discount = self._discount_phrase(
                plan.listed_price_cents,
                plan.current_offer_cents,
                plan.shipping_condition or "seller_pays",
            )
            if discount:
                return (
                    f"{buyer}暂时不行，最多先{discount}。"
                    f"{self._selling_point_sentence(plan.selling_point)}"
                )
            return (
                f"{buyer}暂时不行。"
                f"{self._selling_point_sentence(plan.selling_point)}"
            )
        assert _is_cents(plan.counter_offer_cents)
        discount = self._discount_phrase(
            plan.listed_price_cents,
            plan.counter_offer_cents,
            plan.shipping_condition or "seller_pays",
        )
        if discount:
            return (
                f"可以先{discount}。"
                f"{self._selling_point_sentence(plan.selling_point)}"
            )
        return (
            "当前这个价格可以。"
            f"{self._selling_point_sentence(plan.selling_point)}"
        )

    def _discount_phrase(
        self,
        listed_price_cents: int | None,
        offer_cents: int,
        shipping_condition: ShippingCondition,
    ) -> str:
        if not _is_cents(listed_price_cents) or offer_cents >= listed_price_cents:
            return ""
        discount = listed_price_cents - offer_cents
        label = "包邮" if shipping_condition == "seller_pays" else "不包邮"
        return f"比标价少{self._format_plain_yuan(discount)}元{label}"

    def _discount_amount_phrase(
        self,
        listed_price_cents: int,
        offer_cents: int,
    ) -> str:
        discount = max(listed_price_cents - offer_cents, 0)
        return f"比标价少{self._format_plain_yuan(discount)}元"

    def _selling_point(self, item: Mapping[str, object]) -> str:
        facts = self._facts(item)
        included = facts.get("included_items")
        if isinstance(included, list) and any(
            isinstance(value, str) and "镜头" in value for value in included
        ):
            return "整套机带镜头一起出，性价比已经挺高了。"
        lens = facts.get("lens")
        if isinstance(lens, Mapping) and self._known_text(lens.get("model")):
            return "整套机带镜头一起出，性价比已经挺高了。"
        return "这个价格性价比已经挺高了。"

    @staticmethod
    def _selling_point_sentence(value: str | None) -> str:
        return value if isinstance(value, str) and value.strip() else "这个价格性价比已经挺高了。"

    def _legacy_decide(
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
            discount_prefix = re.search(r"(?:便宜|优惠|再少|少|减)\s*$", prefix)
            if discount_prefix is not None and not prefix.rstrip().endswith("多少"):
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

    @staticmethod
    def _format_plain_yuan(cents: int) -> str:
        amount = Decimal(cents) / Decimal(100)
        if amount == amount.to_integral_value():
            return str(int(amount))
        return f"{amount:.2f}"


def _is_cents(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0
