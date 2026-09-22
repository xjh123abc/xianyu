"""Resolve the one seller item a buyer message may safely discuss."""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence

from app.services.item_service import ItemService
from app.services.xianyu.responses import clarification, item_conflict, unavailable


logger = logging.getLogger(__name__)


_EXPLICIT_ITEM_REFERENCE = re.compile(
    r"(?:商品编号|商品ID|item_id)\s*[:：]?\s*[A-Za-z][A-Za-z0-9_-]{2,}"
    r"|(?<![A-Za-z0-9_])[A-Za-z]+(?:[_-][A-Za-z0-9]+){1,}(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_ITEM_CONTEXT_TERMS = (
    "这个商品",
    "这件商品",
    "这个东西",
    "这件",
    "它",
    "配件",
    "成色",
    "瑕疵",
    "磕碰",
    "维修",
    "拆修",
    "改装",
    "标价",
    "售价",
    "多少钱",
    "在售",
    "售出",
    "卖出",
    "有货",
    "accessory",
    "condition",
    "repair",
)


class ItemContextResolver:
    """Choose, fetch, and validate a concrete item without fallback guessing."""

    def __init__(
        self,
        item_service: ItemService,
        get_item_info: Callable[[str], Awaitable[Mapping[str, object]]],
    ) -> None:
        self._item_service = item_service
        self._get_item_info = get_item_info

    def resolve_text_item_ids(self, query: str) -> list[str]:
        """Resolve configured ID/title references only when item handling is needed."""

        return self._item_service.resolve_item_ids(query)

    @staticmethod
    def may_contain_explicit_item_reference(query: str) -> bool:
        """Detect an ID-shaped reference without loading the item snapshot."""

        return bool(_EXPLICIT_ITEM_REFERENCE.search(str(query or "")))

    @staticmethod
    def requires_item_context(query: str) -> bool:
        """Return whether a buyer need cannot be answered without one item."""

        lowered_query = str(query or "").casefold()
        return any(term in lowered_query for term in _ITEM_CONTEXT_TERMS)

    async def resolve(
        self,
        query: str,
        structured_item_id: str | None,
        current_item_id: str | None,
        *,
        text_item_ids: Sequence[str] | None = None,
        defer_to_order_context: bool = False,
    ) -> tuple[Mapping[str, object] | None, dict[str, object] | None]:
        """Resolve and confirm one item without falling back after a bad switch."""

        explicit_item_id = str(structured_item_id or "").strip().upper() or None
        if text_item_ids is None:
            text_item_ids = self.resolve_text_item_ids(query)
        if len(text_item_ids) > 1:
            return None, item_conflict(
                query,
                "你这条消息里提到不止一件商品，想问的是哪一件？",
            )
        text_item_id = text_item_ids[0] if text_item_ids else None
        if (
            explicit_item_id is not None
            and text_item_id is not None
            and explicit_item_id != text_item_id
        ):
            return None, item_conflict(
                query,
                "你这条消息里的商品信息对不上，想问的是哪一件？",
            )

        candidate_id = explicit_item_id or text_item_id
        has_new_candidate = candidate_id is not None
        if candidate_id is None:
            candidate_id = current_item_id
        if candidate_id is None:
            if defer_to_order_context:
                return None, None
            return None, clarification(query)

        try:
            item = await self._get_item_info(candidate_id)
        except Exception:
            logger.exception("Item MCP lookup failed for item_id=%s", candidate_id)
            response = unavailable(query, "商品信息暂时无法读取。")
            response["reason"] = "item_lookup_failed"
            response["item_id"] = candidate_id
            return None, response
        if not item.get("found"):
            message = (
                "这件商品我暂时没查到，麻烦确认一下。"
                if has_new_candidate
                else "我这边暂时没确认到你问的是哪一件商品。"
            )
            return None, item_conflict(query, message, item_id=candidate_id)
        if not self.valid_item_evidence(item, candidate_id):
            response = unavailable(query, "商品信息暂时无法确认。")
            response["reason"] = "item_evidence_invalid"
            response["item_id"] = candidate_id
            return None, response
        return item, None

    @staticmethod
    def valid_item_evidence(
        item: Mapping[str, object], expected_item_id: str
    ) -> bool:
        """Accept only complete public MCP facts for the requested item."""

        price = item.get("listed_price_cents")
        return (
            item.get("found") is True
            and item.get("item_id") == expected_item_id
            and isinstance(item.get("title"), str)
            and bool(str(item.get("title")).strip())
            and isinstance(price, int)
            and not isinstance(price, bool)
            and price >= 0
            and item.get("sale_status") in {"listed", "sold", "unknown"}
            and item.get("data_source") == "seller_manual"
            and isinstance(item.get("updated_at"), str)
            and bool(str(item.get("updated_at")).strip())
            and ("facts" not in item or isinstance(item.get("facts"), Mapping))
        )
