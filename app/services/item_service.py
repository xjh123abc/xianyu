"""Read-only public item facts for the local Xianyu catalog."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from config.settings import settings


SALE_STATUSES = frozenset({"listed", "sold", "unknown"})
PUBLIC_ITEM_FIELDS = (
    "item_id",
    "title",
    "listed_price_cents",
    "sale_status",
    "data_source",
    "updated_at",
)


def _optional_text(value: object, *, field: str, index: int) -> str | None:
    """Validate a seller-maintained optional text fact without inventing one."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Xianyu item at index {index} has an invalid {field}")
    normalized = value.strip()
    return normalized or None


def _optional_text_list(value: object, *, field: str, index: int) -> list[str] | None:
    """Validate a public list fact, preserving an explicit empty list."""

    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"Xianyu item at index {index} has an invalid {field}")
    values: list[str] = []
    for position, entry in enumerate(value):
        text = _optional_text(entry, field=f"{field}[{position}]", index=index)
        if text is None:
            raise ValueError(f"Xianyu item at index {index} has an empty {field} entry")
        values.append(text)
    return values


def _optional_text_or_unknown(value: object, *, field: str, index: int) -> str | None:
    """Validate an explicitly supplied text fact, including the ``unknown`` sentinel."""

    return _optional_text(value, field=field, index=index)


def _optional_text_list_or_unknown(
    value: object, *, field: str, index: int
) -> list[str] | str | None:
    """Keep a seller-provided list distinct from an explicit unknown value."""

    if value == "unknown":
        return "unknown"
    return _optional_text_list(value, field=field, index=index)


def _validate_lens(value: object, index: int) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"Xianyu item at index {index} has an invalid lens")

    included = value.get("included")
    if included is not None and not isinstance(included, bool):
        raise ValueError(f"Xianyu item at index {index} has an invalid lens.included")
    focal_length = value.get("focal_length_mm")
    if focal_length is not None and (
        not isinstance(focal_length, int)
        or isinstance(focal_length, bool)
        or focal_length <= 0
    ):
        raise ValueError(
            f"Xianyu item at index {index} has an invalid lens.focal_length_mm"
        )
    return {
        "included": included,
        "focal_length_mm": focal_length,
        "model": _optional_text(value.get("model"), field="lens.model", index=index),
    }


def _validate_condition(value: object, index: int) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"Xianyu item at index {index} has an invalid condition")
    return {
        "appearance": _optional_text(value.get("appearance"), field="condition.appearance", index=index),
        "function": _optional_text(value.get("function"), field="condition.function", index=index),
        "known_issues": _optional_text_list_or_unknown(
            value.get("known_issues"), field="condition.known_issues", index=index
        ),
        "summary": _optional_text_or_unknown(
            value.get("summary"), field="condition.summary", index=index
        ),
        "scratches": _optional_text_or_unknown(
            value.get("scratches"), field="condition.scratches", index=index
        ),
        "dents": _optional_text_or_unknown(
            value.get("dents"), field="condition.dents", index=index
        ),
    }


def _validate_text_mapping(
    value: object,
    *,
    field: str,
    keys: tuple[str, ...],
    index: int,
) -> dict[str, str | None] | None:
    """Validate a fixed seller-fact object without filling in missing values."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"Xianyu item at index {index} has an invalid {field}")
    return {
        key: _optional_text_or_unknown(value.get(key), field=f"{field}.{key}", index=index)
        for key in keys
    }


def _validate_identity(raw_item: Mapping[str, object], index: int) -> dict[str, object] | None:
    identity_fields = ("category", "brand", "model")
    if not any(field in raw_item for field in identity_fields):
        return None
    return {
        field: _optional_text(raw_item.get(field), field=field, index=index)
        for field in identity_fields
    }


def _validate_structured_facts(raw_item: Mapping[str, object], index: int) -> dict[str, object]:
    """Extract explicitly supplied product facts from seller-maintained data.

    Absence means the seller has not supplied that fact.  This distinction lets
    the chat layer fall back to scoped knowledge, while an explicit value is
    safe to answer deterministically without an LLM.
    """

    facts: dict[str, object] = {}
    identity = _validate_identity(raw_item, index)
    if identity is not None:
        facts["identity"] = identity
    if "lens" in raw_item:
        facts["lens"] = _validate_lens(raw_item.get("lens"), index)
    if "condition" in raw_item:
        facts["condition"] = _validate_condition(raw_item.get("condition"), index)
    if "negotiation" in raw_item:
        facts["negotiation"] = _optional_text_or_unknown(
            raw_item.get("negotiation"), field="negotiation", index=index
        )
    if "function" in raw_item:
        facts["function"] = _validate_text_mapping(
            raw_item.get("function"),
            field="function",
            keys=("overall", "shutter"),
            index=index,
        )
    if "history" in raw_item:
        facts["history"] = _validate_text_mapping(
            raw_item.get("history"),
            field="history",
            keys=("repair_history", "drop_history", "disassembly_history"),
            index=index,
        )
    if "inspection_record" in raw_item:
        facts["inspection_record"] = _validate_text_mapping(
            raw_item.get("inspection_record"),
            field="inspection_record",
            keys=("general", "meter_phone_comparison"),
            index=index,
        )
    if "accessories" in raw_item:
        facts["accessories"] = _optional_text_list_or_unknown(
            raw_item.get("accessories"), field="accessories", index=index
        )
    if "accessory_details" in raw_item:
        facts["accessory_details"] = _validate_text_mapping(
            raw_item.get("accessory_details"),
            field="accessory_details",
            keys=("completeness", "original_accessories", "image_items", "manual_or_packaging"),
            index=index,
        )
    if "shipping" in raw_item:
        facts["shipping"] = _validate_text_mapping(
            raw_item.get("shipping"),
            field="shipping",
            keys=(
                "ship_from",
                "carrier",
                "shipping_fee",
                "dispatch_time",
                "negotiation_policy",
                "negotiation_express_policy",
            ),
            index=index,
        )
    if "product_info" in raw_item:
        facts["product_info"] = _validate_text_mapping(
            raw_item.get("product_info"),
            field="product_info",
            keys=("production_year", "beginner_suitability", "usage", "sale_reason"),
            index=index,
        )
    if "after_sale" in raw_item:
        facts["after_sale"] = _validate_text_mapping(
            raw_item.get("after_sale"),
            field="after_sale",
            keys=("transaction_channel", "return_policy", "description_policy", "inspection_confirmation"),
            index=index,
        )
    if "included_items" in raw_item:
        facts["included_items"] = _optional_text_list(
            raw_item.get("included_items"), field="included_items", index=index
        )
    if "listing_description" in raw_item:
        facts["listing_description"] = _optional_text(
            raw_item.get("listing_description"), field="listing_description", index=index
        )
    if "fact_conflicts" in raw_item:
        conflicts = _optional_text_list(
            raw_item.get("fact_conflicts"), field="fact_conflicts", index=index
        )
        allowed_conflicts = {
            "identity", "lens", "condition", "included_items", "accessories",
            "function", "history", "inspection_record", "shipping", "product_info", "after_sale", "negotiation",
        }
        invalid = set(conflicts or []) - allowed_conflicts
        if invalid:
            raise ValueError(
                f"Xianyu item at index {index} has unsupported fact_conflicts: "
                + ", ".join(sorted(invalid))
            )
        facts["fact_conflicts"] = conflicts or []
    return facts
_ITEM_ID_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:DEMO_ITEM_[A-Za-z0-9_-]+|TEST_[A-Za-z0-9_-]+)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_LABELED_ITEM_ID_PATTERN = re.compile(
    r"(?:商品编号|商品ID|item_id)\s*[:：]?\s*([A-Za-z][A-Za-z0-9_-]{2,})",
    re.IGNORECASE,
)


def configured_items_path() -> Path:
    """Resolve the configured seller-maintained item snapshot."""

    if settings is None:
        raise RuntimeError("Project settings are unavailable")
    path = Path(settings.xianyu_items_path)
    if not path.is_absolute():
        path = Path(__file__).parents[2] / path
    return path.resolve()


class ItemService:
    """Load and validate only public item facts from the seller snapshot."""

    def __init__(self, items_path: str | Path | None = None) -> None:
        self.items_path = (
            Path(items_path).resolve() if items_path is not None else configured_items_path()
        )
        self.platform_items_path = self.items_path.parent / "platform_items.json"

    def list_items(self) -> list[dict[str, Any]]:
        """Return all validated public item records in file order."""

        if not self.items_path.is_file():
            raise FileNotFoundError(f"Xianyu item snapshot not found: {self.items_path}")
        try:
            raw_items = json.loads(self.items_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Invalid Xianyu item snapshot: {self.items_path}") from error
        if not isinstance(raw_items, list):
            raise ValueError("Xianyu item snapshot must contain a JSON array")
        items = [self._validate_item(item, index) for index, item in enumerate(raw_items)]
        return self._merge_platform_snapshot(items)

    def _merge_platform_snapshot(
        self, items: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Overlay only public listing fields; retain seller policy and condition facts."""
        if not self.platform_items_path.is_file():
            return items
        try:
            payload = json.loads(self.platform_items_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("Invalid Xianyu platform item snapshot") from error
        entries = payload.get("items") if isinstance(payload, Mapping) else None
        if not isinstance(entries, list):
            raise ValueError("Xianyu platform item snapshot must contain an items array")
        snapshots_by_item_id: dict[str, list[Mapping[str, object]]] = {}
        for entry in entries:
            if isinstance(entry, Mapping) and isinstance(entry.get("item_id"), str):
                item_id = str(entry["item_id"]).strip().upper()
                snapshots_by_item_id.setdefault(item_id, []).append(entry)

        by_item_id: dict[str, Mapping[str, object]] = {}
        for item_id, snapshots in snapshots_by_item_id.items():
            # Multiple seller accounts can sync the same Xianyu listing. That
            # is still one listing; snapshots for different platform IDs remain
            # ambiguous and must not override the seller-maintained record.
            platform_ids = {
                str(snapshot.get("platform_item_id", "")).strip()
                for snapshot in snapshots
            }
            if len(platform_ids) != 1:
                continue
            platform_id = next(iter(platform_ids))
            if not platform_id:
                continue
            by_item_id[item_id] = max(
                enumerate(snapshots),
                key=lambda pair: (
                    str(pair[1].get("synced_at", "")),
                    pair[0],
                ),
            )[1]

        for item in items:
            snapshot = by_item_id.get(str(item["item_id"]).upper())
            if snapshot is None:
                continue
            title = snapshot.get("title")
            description = snapshot.get("listing_description")
            if isinstance(title, str) and title.strip():
                item["title"] = title.strip()
            if isinstance(description, str):
                item["facts"]["listing_description"] = description.strip() or None
            status = snapshot.get("sale_status")
            if isinstance(status, str) and status.strip().lower() in {"listed", "sold"}:
                item["sale_status"] = status.strip().lower()
            # A failed or unrecognized platform status refresh is not evidence
            # that a seller-confirmed status changed. Keep the validated
            # seller-maintained status (or its explicit "unknown" value).
            platform_price = snapshot.get("listed_price_cents")
            if (
                isinstance(platform_price, int)
                and not isinstance(platform_price, bool)
                and platform_price >= 0
            ):
                if platform_price != item["listed_price_cents"]:
                    conflicts = set(item["facts"].get("fact_conflicts", []))
                    conflicts.add("listed_price_cents")
                    item["facts"]["fact_conflicts"] = sorted(conflicts)
                item["listed_price_cents"] = platform_price
            synced_at = snapshot.get("synced_at")
            if isinstance(synced_at, str) and synced_at.strip():
                item["updated_at"] = synced_at
            item["data_source"] = "seller_snapshot+xianyu_platform"
            item["facts"]["platform_listing"] = {
                "platform_item_id": snapshot.get("platform_item_id"),
                "synced_at": synced_at,
                "price_conflict": "listed_price_cents"
                in item["facts"].get("fact_conflicts", []),
            }
        return items

    def get_item_info(self, item_id: str) -> dict[str, Any]:
        """Return one public item record, or a stable not-found response."""

        normalized_id = self._normalize_id(item_id)
        for item in self.list_items():
            if item["item_id"] == normalized_id:
                return {
                    "found": True,
                    **{key: item[key] for key in PUBLIC_ITEM_FIELDS},
                    "facts": item["facts"],
                }
        return {
            "found": False,
            "item_id": normalized_id,
            "title": None,
            "listed_price_cents": None,
            "sale_status": None,
            "data_source": None,
            "updated_at": None,
            "facts": {},
        }

    def resolve_item_ids(self, text: str) -> list[str]:
        """Return every configured item explicitly identified by ID or title."""

        raw_text = str(text or "").strip()
        normalized_text = raw_text.casefold()
        if not normalized_text:
            return []

        matches = [match.group(0).upper() for match in _ITEM_ID_PATTERN.finditer(raw_text)]
        matches.extend(
            match.group(1).upper() for match in _LABELED_ITEM_ID_PATTERN.finditer(raw_text)
        )
        for item in self.list_items():
            if item["item_id"].casefold() in normalized_text:
                matches.append(item["item_id"])
            if item["title"].casefold() in normalized_text:
                matches.append(item["item_id"])
        return list(dict.fromkeys(matches))

    @staticmethod
    def _normalize_id(item_id: str) -> str:
        normalized_id = str(item_id or "").strip().upper()
        if not normalized_id:
            raise ValueError("item_id must not be empty")
        return normalized_id

    @staticmethod
    def _validate_item(raw_item: object, index: int) -> dict[str, Any]:
        if not isinstance(raw_item, Mapping):
            raise ValueError(f"Xianyu item at index {index} must be an object")
        missing = [field for field in PUBLIC_ITEM_FIELDS if field not in raw_item]
        if missing:
            raise ValueError(f"Xianyu item at index {index} is missing: {', '.join(missing)}")

        item_id = str(raw_item["item_id"]).strip()
        title = str(raw_item["title"]).strip()
        if not item_id or not title:
            raise ValueError(f"Xianyu item at index {index} has an empty ID or title")
        try:
            price_cents = int(raw_item["listed_price_cents"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Xianyu item at index {index} has an invalid price") from error
        if price_cents < 0:
            raise ValueError(f"Xianyu item at index {index} has a negative price")
        sale_status = str(raw_item["sale_status"]).strip().lower()
        if sale_status not in SALE_STATUSES:
            raise ValueError(f"Xianyu item at index {index} has an invalid sale_status")
        data_source = str(raw_item["data_source"]).strip()
        updated_at = str(raw_item["updated_at"]).strip()
        if not data_source or not updated_at:
            raise ValueError(f"Xianyu item at index {index} has missing source metadata")
        return {
            "item_id": item_id,
            "title": title,
            "listed_price_cents": price_cents,
            "sale_status": sale_status,
            "data_source": data_source,
            "updated_at": updated_at,
            "facts": _validate_structured_facts(raw_item, index),
        }
