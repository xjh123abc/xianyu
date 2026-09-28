"""Refresh the local public snapshot for explicitly bound Xianyu listings."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Protocol

from app.channels.xianyu.store import ChannelStore


class ItemInfoClient(Protocol):
    def get_item_info(self, item_id: str) -> object: ...


def _item_do(payload: object) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("platform item response is not an object")
    data = payload.get("data")
    item = data.get("itemDO") if isinstance(data, Mapping) else None
    if not isinstance(item, Mapping):
        raise ValueError("platform item response has no item data")
    return item


def _status(item: Mapping[str, Any]) -> str:
    raw = item.get("itemStatus", item.get("status"))
    if not isinstance(raw, str):
        return "unknown"
    normalized = raw.strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized in {"listed", "on_sale", "onsale", "active", "published", "selling"}:
        return "listed"
    if normalized in {"sold", "closed", "sold_out"}:
        return "sold"
    return "unknown"


def _price_cents(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    if not amount.is_finite() or amount < 0 or amount != amount.to_integral_value():
        return None
    # The pinned Xianyu client formats soldPrice by dividing it by 100.
    return int(amount)


def _normalise_listing(platform_item_id: str, item: Mapping[str, Any], synced_at: str) -> dict[str, object]:
    title = item.get("title")
    description = item.get("desc")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("platform item has no usable title")
    if description is not None and not isinstance(description, str):
        raise ValueError("platform item description has an invalid type")
    price = _price_cents(item.get("soldPrice"))
    if price is None:
        raise ValueError("platform item has no usable price")
    return {
        "platform_item_id": platform_item_id,
        "title": title.strip(),
        "listing_description": description.strip() if isinstance(description, str) else "",
        "listed_price_cents": price,
        "sale_status": _status(item),
        "data_source": "xianyu_platform",
        "synced_at": synced_at,
    }


def sync_bound_items(
    api: ItemInfoClient,
    store: ChannelStore,
    *,
    account_id: str,
    snapshot_path: str | Path,
) -> dict[str, object]:
    """Fetch and atomically persist current public data for bound listings only."""
    path = Path(snapshot_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    previous: dict[tuple[str, str], dict[str, object]] = {}
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            entries = payload.get("items", []) if isinstance(payload, Mapping) else []
            if isinstance(entries, list):
                for entry in entries:
                    if isinstance(entry, Mapping):
                        key = (str(entry.get("account_id", "")), str(entry.get("platform_item_id", "")))
                        if all(key):
                            previous[key] = dict(entry)
        except (OSError, json.JSONDecodeError):
            # A corrupt old cache is never overwritten with a partial sync.
            raise RuntimeError("existing platform item snapshot is unreadable") from None

    synced_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    bindings = store.list_bound_items(account_id)
    active_platform_ids = {binding["platform_item_id"] for binding in bindings}
    removed = [
        key for key in previous
        if key[0] == account_id and key[1] not in active_platform_ids
    ]
    for key in removed:
        del previous[key]

    succeeded = 0
    failures: list[dict[str, str]] = []
    for binding in bindings:
        platform_id = binding["platform_item_id"]
        try:
            raw = api.get_item_info(platform_id)
            listing = _normalise_listing(platform_id, _item_do(raw), synced_at)
            previous[(account_id, platform_id)] = {
                "account_id": account_id,
                "item_id": binding["item_id"],
                **listing,
            }
            succeeded += 1
        except Exception as error:
            failures.append({"platform_item_id": platform_id, "error": type(error).__name__})

    if succeeded or removed:
        serialised = {
            "version": 1,
            "items": [previous[key] for key in sorted(previous)],
        }
        descriptor, temporary_name = tempfile.mkstemp(prefix="platform_items_", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(serialised, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    return {
        "bound": len(bindings),
        "synced": succeeded,
        "failed": failures,
        "snapshot_path": str(path),
    }
