"""Private, validated price policies for continuous Xianyu bargaining.

The policy file deliberately lives outside item facts.  It may contain an
authorised floor and therefore must never be returned by item APIs or RAG.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from config.settings import settings


class NegotiationPolicyError(ValueError):
    """Raised when a private policy cannot safely be used."""


@dataclass(frozen=True, slots=True)
class ShippingPricePolicy:
    offers_cents: tuple[int, ...]
    floor_cents: int


@dataclass(frozen=True, slots=True)
class NegotiationPolicy:
    item_id: str
    version: int
    listed_price_cents: int
    seller_pays: ShippingPricePolicy
    buyer_pays: ShippingPricePolicy

    def for_shipping(self, condition: str) -> ShippingPricePolicy:
        if condition == "seller_pays":
            return self.seller_pays
        if condition == "buyer_pays":
            return self.buyer_pays
        raise NegotiationPolicyError("shipping_condition_invalid")


class NegotiationPolicyStore:
    """Load one small JSON policy file with strict, fail-closed validation."""

    def __init__(self, path: str | Path | None = None) -> None:
        configured_path = (
            Path(settings.xianyu_negotiation_policy_path)
            if settings is not None
            else Path("data/xianyu/negotiation_policies.json")
        )
        if not configured_path.is_absolute():
            configured_path = Path(__file__).resolve().parents[2] / configured_path
        self.path = Path(path) if path is not None else configured_path
        self._cached_mtime_ns: int | None = None
        self._cached: dict[str, NegotiationPolicy] = {}

    def get(self, item: Mapping[str, object]) -> NegotiationPolicy | None:
        item_id = item.get("item_id")
        if not isinstance(item_id, str) or not item_id.strip():
            return None
        policy = self._load().get(item_id.strip().upper())
        if policy is None:
            return None
        listed_price = item.get("listed_price_cents")
        if not _cents(listed_price):
            raise NegotiationPolicyError("listed_price_unavailable")
        if listed_price != policy.listed_price_cents:
            raise NegotiationPolicyError("negotiation_policy_listed_price_conflict")
        return policy

    def version_matches(self, item: Mapping[str, object], version: object) -> bool:
        try:
            policy = self.get(item)
        except NegotiationPolicyError:
            return False
        return policy is not None and policy.version == version

    def _load(self) -> dict[str, NegotiationPolicy]:
        if not self.path.exists():
            return {}
        stat = self.path.stat()
        if self._cached_mtime_ns == stat.st_mtime_ns:
            return self._cached
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise NegotiationPolicyError("negotiation_policy_file_invalid") from exc
        if not isinstance(raw, Mapping):
            raise NegotiationPolicyError("negotiation_policy_file_invalid")
        loaded: dict[str, NegotiationPolicy] = {}
        for raw_item_id, raw_policy in raw.items():
            if not isinstance(raw_item_id, str) or not raw_item_id.strip():
                raise NegotiationPolicyError("negotiation_policy_item_id_invalid")
            loaded[raw_item_id.strip().upper()] = _parse_policy(
                raw_item_id.strip().upper(), raw_policy
            )
        self._cached_mtime_ns = stat.st_mtime_ns
        self._cached = loaded
        return loaded


def _parse_policy(item_id: str, raw: object) -> NegotiationPolicy:
    if not isinstance(raw, Mapping):
        raise NegotiationPolicyError("negotiation_policy_invalid")
    version = raw.get("version")
    listed_price = raw.get("listed_price_cents")
    if not _non_negative_int(version) or not _cents(listed_price):
        raise NegotiationPolicyError("negotiation_policy_invalid")
    seller_pays = _parse_shipping_policy(raw.get("seller_pays"))
    buyer_pays = _parse_shipping_policy(raw.get("buyer_pays"))
    if any(
        offer > listed_price
        for shipping in (seller_pays, buyer_pays)
        for offer in shipping.offers_cents
    ):
        raise NegotiationPolicyError("negotiation_offer_exceeds_listed_price")
    return NegotiationPolicy(
        item_id=item_id,
        version=version,
        listed_price_cents=listed_price,
        seller_pays=seller_pays,
        buyer_pays=buyer_pays,
    )


def _parse_shipping_policy(raw: object) -> ShippingPricePolicy:
    if not isinstance(raw, Mapping):
        raise NegotiationPolicyError("negotiation_shipping_policy_invalid")
    floor = raw.get("floor_cents")
    offers = raw.get("offers_cents")
    if not _cents(floor) or not isinstance(offers, Sequence) or isinstance(offers, (str, bytes)):
        raise NegotiationPolicyError("negotiation_shipping_policy_invalid")
    normalized = tuple(offers)
    if not normalized or any(not _cents(offer) for offer in normalized):
        raise NegotiationPolicyError("negotiation_shipping_policy_invalid")
    if any(left < right for left, right in zip(normalized, normalized[1:])):
        raise NegotiationPolicyError("negotiation_offer_tiers_increase")
    if any(offer < floor for offer in normalized):
        raise NegotiationPolicyError("negotiation_offer_below_floor")
    return ShippingPricePolicy(offers_cents=normalized, floor_cents=floor)


def _non_negative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _cents(value: object) -> bool:
    return _non_negative_int(value)
