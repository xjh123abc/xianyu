"""Unified evidence-retrieval boundary over the existing RAG services."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Literal

from app.services.rag_service import RAGService
from app.services.technical_knowledge_service import TechnicalKnowledgeService


KnowledgeScope = Literal["merchant", "product", "platform", "item"]
_SCOPES = frozenset({"merchant", "product", "platform", "item"})
RagProvider = Callable[[], RAGService]


class KnowledgeService:
    """Retrieve evidence without exposing retrieval implementation details."""

    def __init__(self, rag_service: RagProvider) -> None:
        self._rag_service = rag_service

    def warm_up(self) -> None:
        """Preserve the existing lazy model-loading behaviour of RAGService."""

        self._rag_service().warm_up()

    def search(
        self,
        query: str,
        scope: KnowledgeScope,
        *,
        platform: str | None = None,
        product_model: str | None = None,
        item_id: str | None = None,
    ) -> dict[str, object]:
        """Return existing ``RAGService.prepare`` evidence through one contract."""

        if scope not in _SCOPES:
            raise ValueError("scope must be merchant, product, platform, or item")
        normalized_query = str(query or "").strip()
        if not normalized_query:
            raise ValueError("query must be a non-empty string")
        normalized_item_id = str(item_id or "").strip() or None
        if scope == "item" and normalized_item_id is None:
            raise ValueError("item scope requires item_id")

        prepared = self._rag_service().prepare(
            normalized_query,
            item_id=normalized_item_id if scope == "item" else None,
        )
        return {
            **prepared,
            "knowledge_scope": scope,
            "platform": str(platform or "").strip() or None,
            "product_model": str(product_model or "").strip() or None,
        }

    def search_model_knowledge(
        self,
        query: str,
        *,
        item: Mapping[str, object],
        local_prepared: Mapping[str, object] | None = None,
        technical_service: TechnicalKnowledgeService | None = None,
    ) -> dict[str, object]:
        """Return local model evidence or a controlled web-search fallback."""

        prepared = dict(local_prepared) if local_prepared is not None else self.search(
            query,
            "item",
            platform="xianyu",
            item_id=str(item.get("item_id", "")),
        )
        context = prepared.get("context")
        context_text = (
            str(context.get("context", "")).strip()
            if isinstance(context, Mapping)
            else ""
        )
        if context_text:
            return prepared
        service = technical_service or TechnicalKnowledgeService()
        return service.prepare_model_evidence(query, item=item)
