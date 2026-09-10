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
_ITEM_ID_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])DEMO_ITEM_[A-Za-z0-9_-]+(?![A-Za-z0-9_])",
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
        return [self._validate_item(item, index) for index, item in enumerate(raw_items)]

    def get_item_info(self, item_id: str) -> dict[str, Any]:
        """Return one public item record, or a stable not-found response."""

        normalized_id = self._normalize_id(item_id)
        for item in self.list_items():
            if item["item_id"] == normalized_id:
                return {
                    "found": True,
                    **{key: item[key] for key in PUBLIC_ITEM_FIELDS},
                }
        return {
            "found": False,
            "item_id": normalized_id,
            "title": None,
            "listed_price_cents": None,
            "sale_status": None,
            "data_source": None,
            "updated_at": None,
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
        }
