"""Scoped Xianyu knowledge retrieval and grounded answer generation."""

from __future__ import annotations

import asyncio
import logging
import re
import warnings
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from app.generation.deepseek import DeepSeekGenerator
from app.services.intent_router import IntentMatch
from app.services.query_planner import (
    COMMON_KNOWLEDGE_RETRIEVAL_HINT,
    QuestionPlan,
)
from app.services.knowledge_service import KnowledgeService
from app.services.xianyu.item_fact_responder import ItemFactResponder
from app.services.xianyu.responses import common_handoff, valid_knowledge_sources


logger = logging.getLogger(__name__)

LegacyItemHandler = Callable[
    [str, Mapping[str, object], Sequence[Mapping[str, Any]] | None],
    Awaitable[dict[str, object]],
]


class XianyuKnowledgeResponder:
    """Retrieve only the evidence scope required by an item/seller question."""

    def __init__(
        self,
        *,
        knowledge_service: KnowledgeService,
        generator: Callable[[], DeepSeekGenerator],
        fact_responder: ItemFactResponder,
        route_intent: Callable[[str], IntentMatch],
        legacy_item_handler: LegacyItemHandler | None = None,
    ) -> None:
        # Keep these arguments in the constructor while external compatibility
        # callers migrate; item fact decisions now belong to expert agents.
        del fact_responder, route_intent
        self._knowledge_service = knowledge_service
        self._generator = generator
        self._legacy_item_handler = legacy_item_handler

    async def handle_item(
        self,
        query: str,
        item: Mapping[str, object],
        *,
        plan: QuestionPlan | None = None,
        history: Sequence[Mapping[str, Any]] | None = None,
        intent_match: IntentMatch | None = None,
        force_full_item_answer: bool = False,
    ) -> dict[str, object]:
        """Deprecated adapter to the expert chain; do not use in new code."""

        warnings.warn(
            "XianyuKnowledgeResponder.handle_item() is deprecated; "
            "plan and execute Task values through XianyuExpertOrchestrator instead",
            DeprecationWarning,
            stacklevel=2,
        )
        del plan, intent_match, force_full_item_answer
        if self._legacy_item_handler is None:
            raise RuntimeError("legacy item compatibility handler is not configured")
        return await self._legacy_item_handler(query, item, history)

    async def handle_common(
        self,
        query: str,
    ) -> dict[str, object]:
        """Answer an item-independent question from common seller rules only."""

        prepared: Mapping[str, object] | None = None
        try:
            retrieval_query = f"{query} {COMMON_KNOWLEDGE_RETRIEVAL_HINT}".strip()
            self._knowledge_service.warm_up()
            prepared = await asyncio.to_thread(
                self._knowledge_service.search,
                retrieval_query,
                "merchant",
                platform="xianyu",
            )
            context = prepared.get("context")
            knowledge_sources = valid_knowledge_sources(prepared)
            if (
                not prepared.get("can_answer")
                or not isinstance(context, Mapping)
                or not knowledge_sources
            ):
                return self._common_handoff(
                    query,
                    "common_knowledge_unavailable",
                )
            context_text = str(context.get("context", "")).strip()
            if not context_text:
                raise RuntimeError("empty common Xianyu context")
            answer = self._generator().generate_xianyu(
                query,
                {},
                "卖家通用规则：\n" + context_text,
            )
            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError("empty common Xianyu answer")
            return {
                "query": query,
                "route": "xianyu",
                "action": "reply",
                "answer": answer.strip(),
                "sources": knowledge_sources,
                "results": prepared.get("results", []),
                "reliability": prepared.get("reliability"),
                "next_step": None,
                "can_answer": True,
            }
        except Exception:
            logger.exception("Common Xianyu RAG failed")
            has_prepared_evidence = prepared is not None and bool(
                valid_knowledge_sources(prepared)
            )
            response = self._common_handoff(
                query,
                "common_knowledge_generation_failed"
                if has_prepared_evidence
                else "common_knowledge_unavailable",
            )
            if has_prepared_evidence and prepared is not None:
                response.update(
                    {
                        "sources": valid_knowledge_sources(prepared),
                        "results": prepared.get("results", []),
                        "reliability": prepared.get("reliability"),
                    }
                )
            return response

    async def prepare_evidence(
        self,
        question: str,
        *,
        item: Mapping[str, object] | None,
        scope: str,
    ) -> dict[str, object]:
        """Prepare scoped evidence for one expert task without generating a reply.

        The returned mapping intentionally preserves the existing RAG evidence
        shape and adds only ``issues``.  Product tasks may search the selected
        item's scope; seller-rule tasks may search only the common scope.
        """

        if scope not in {"item", "common"}:
            bundle = self._knowledge_bundle([], [], [], None)
            bundle["issues"] = ["knowledge_scope_unavailable"]
            return bundle
        if scope == "item" and item is None:
            bundle = self._knowledge_bundle([], [], [], None)
            bundle["issues"] = ["item_context_unavailable"]
            return bundle
        collector_item = item or {"item_id": "common", "title": ""}
        prepared = await self._prepare_evidence_needs(
            ({"question": question, "scope": scope},),
            collector_item,
        )
        return prepared

    async def _prepare_evidence_needs(
        self,
        needs: Sequence[Mapping[str, str]],
        item: Mapping[str, object],
    ) -> dict[str, object]:
        """Collect scoped evidence for expert tasks."""

        contexts: list[str] = []
        sources: list[Mapping[str, object]] = []
        results: list[object] = []
        issues: list[str] = []
        reliability: object = None
        try:
            self._knowledge_service.warm_up()
        except Exception:
            logger.exception("Xianyu RAG warm-up failed for item_id=%s", item["item_id"])
            issues.append("knowledge_warmup_failed")
            bundle = self._knowledge_bundle(contexts, sources, results, reliability)
            bundle["issues"] = issues
            return bundle

        for need in needs:
            retrieval_query = self.retrieval_query(
                need["question"],
                item_title=str(item["title"]) if need["scope"] == "item" else "",
            )
            if need["scope"] == "common":
                retrieval_query = (
                    f"{retrieval_query} {COMMON_KNOWLEDGE_RETRIEVAL_HINT}"
                ).strip()
            try:
                prepared = await asyncio.to_thread(
                    self._knowledge_service.search,
                    retrieval_query,
                    "item" if need["scope"] == "item" else "merchant",
                    platform="xianyu",
                    item_id=str(item["item_id"])
                    if need["scope"] == "item"
                    else None,
                )
            except Exception:
                logger.exception(
                    "Xianyu RAG preparation failed for question=%s item_id=%s",
                    need["question"],
                    item["item_id"],
                )
                issues.append("knowledge_retrieval_failed")
                continue

            prepared_results = prepared.get("results")
            if isinstance(prepared_results, list):
                results.extend(prepared_results)
            current_reliability = prepared.get("reliability")
            if reliability is None or not prepared.get("can_answer"):
                reliability = current_reliability
            context = prepared.get("context")
            current_sources = valid_knowledge_sources(prepared)
            context_text = (
                str(context.get("context", "")).strip()
                if isinstance(context, Mapping)
                else ""
            )
            if not prepared.get("can_answer") or not context_text or not current_sources:
                issues.append("knowledge_evidence_unavailable")
                continue

            contexts.append(f"[{need['scope']}] {need['question']}\n{context_text}")
            for source in current_sources:
                if source not in sources:
                    sources.append(source)

        bundle = self._knowledge_bundle(contexts, sources, results, reliability)
        bundle["issues"] = issues
        return bundle

    @staticmethod
    def _knowledge_bundle(
        contexts: Sequence[str],
        sources: list[Mapping[str, object]],
        results: list[object],
        reliability: object,
    ) -> dict[str, object]:
        context_text = "\n\n".join(contexts)
        return {
            "can_answer": bool(contexts),
            "context": {"context": context_text, "sources": sources}
            if context_text
            else None,
            "sources": sources,
            "results": results,
            "reliability": reliability,
        }

    @staticmethod
    def retrieval_query(
        query: str,
        *,
        item_title: str,
    ) -> str:
        """Keep structured fact wording from weakening knowledge retrieval."""

        cleaned = re.sub(r"(?:这个|这件)?商品", " ", query)
        cleaned = re.sub(r"[\s，,。.!！?？、；;：:]+", " ", cleaned).strip()
        cleaned = re.sub(r"^(?:和|及|与)\s*", "", cleaned).strip()
        normalized_title = item_title.strip()
        if normalized_title and normalized_title.casefold() not in cleaned.casefold():
            cleaned = f"{normalized_title} {cleaned}".strip()
        return cleaned or query

    @staticmethod
    def _common_handoff(query: str, answer: str) -> dict[str, object]:
        return common_handoff(query, answer)
