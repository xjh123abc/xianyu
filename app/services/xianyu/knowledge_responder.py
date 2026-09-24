"""Scoped Xianyu knowledge retrieval and grounded answer generation."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from app.generation.deepseek import DeepSeekGenerator
from app.services.query_planner import (
    COMMON_KNOWLEDGE_RETRIEVAL_HINT,
)
from app.services.knowledge_service import KnowledgeService
from app.services.technical_knowledge_service import TechnicalKnowledgeService
from app.services.xianyu.responses import valid_knowledge_sources
from config.paths import resolve_project_path
from config.settings import settings


logger = logging.getLogger(__name__)
_COMMON_RULES_FILE = "seller_rules.md"
_COMMON_RULES_SOURCE = {"source": _COMMON_RULES_FILE, "index": 0}
_COMMON_RULES_BASE_KEYWORDS = (
    "售后",
    "退货",
    "退款",
    "质量问题",
    "描述不符",
    "卖家责任",
    "平台规则",
    "争议",
)
_COMMON_RULES_DAMAGE_KEYWORDS = (
    "到货即损坏",
    "破损",
    "商品破碎",
    "损坏",
    "商品问题",
    "合理证据",
    "照片",
    "视频",
    "包装",
    "面单",
)
_COMMON_RULES_FEE_KEYWORDS = ("运费", "退货运费", "费用")
_COMMON_RULES_SEVEN_DAY_KEYWORDS = ("七日无理由", "七天无理由", "无理由")
_COMMON_RULES_XIANYU_KEYWORDS = ("闲鱼", "个人闲置", "经营性卖家")

class XianyuKnowledgeResponder:
    """Retrieve only the evidence scope required by an item/seller question."""

    def __init__(
        self,
        *,
        knowledge_service: KnowledgeService,
        generator: Callable[[], DeepSeekGenerator],
        technical_knowledge_service: TechnicalKnowledgeService | None = None,
    ) -> None:
        self._knowledge_service = knowledge_service
        self._generator = generator
        self._technical_knowledge_service = technical_knowledge_service

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
            if not isinstance(context, Mapping):
                raise RuntimeError("missing common Xianyu context")
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
                "answer": answer,
                "raw_answer": answer,
                "evidence": context_text,
                "sources": knowledge_sources,
                "results": prepared.get("results", []),
                "reliability": prepared.get("reliability"),
                "next_step": None,
                "can_answer": True,
            }
        except Exception:
            logger.exception("Common Xianyu RAG failed")
            prepared = prepared or _common_rules_fallback(
                query,
                failure_reason="common_knowledge_generation_failed",
            )
            context = prepared.get("context")
            context_text = (
                str(context.get("context", "")).strip()
                if isinstance(context, Mapping)
                else ""
            )
            if context_text:
                try:
                    answer = self._generator().generate_xianyu(
                        query,
                        {},
                        "卖家通用规则：\n" + context_text,
                    )
                    if isinstance(answer, str) and answer.strip():
                        return {
                            "query": query,
                            "route": "xianyu",
                            "action": "reply",
                            "answer": answer,
                            "raw_answer": answer,
                            "evidence": context_text,
                            "sources": valid_knowledge_sources(prepared),
                            "results": prepared.get("results", []),
                            "reliability": prepared.get("reliability"),
                            "next_step": None,
                            "can_answer": True,
                        }
                except Exception:
                    logger.exception("Common Xianyu fallback generation failed")
            return {
                "query": query,
                "route": "xianyu",
                "action": "reply",
                "answer": "",
                "sources": valid_knowledge_sources(prepared or {}),
                "results": (prepared or {}).get("results", []),
                "reliability": (prepared or {}).get("reliability"),
                "next_step": None,
                "can_answer": False,
                "reason": "common_knowledge_generation_failed",
            }

    async def prepare_evidence(
        self,
        question: str,
        *,
        item: Mapping[str, object] | None,
        scope: str,
        allow_web_fallback: bool = False,
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
            allow_web_fallback=allow_web_fallback,
        )
        return prepared

    async def _prepare_evidence_needs(
        self,
        needs: Sequence[Mapping[str, str]],
        item: Mapping[str, object],
        *,
        allow_web_fallback: bool = False,
    ) -> dict[str, object]:
        """Collect scoped evidence for expert tasks."""

        contexts: list[str] = []
        sources: list[Mapping[str, object]] = []
        results: list[object] = []
        issues: list[str] = []
        reliability: object = None
        local_retrieval_available = True
        try:
            self._knowledge_service.warm_up()
        except Exception:
            logger.exception("Xianyu RAG warm-up failed for item_id=%s", item["item_id"])
            issues.append("knowledge_warmup_failed")
            has_common_need = any(need.get("scope") == "common" for need in needs)
            if not allow_web_fallback and not has_common_need:
                bundle = self._knowledge_bundle(contexts, sources, results, reliability)
                bundle["issues"] = issues
                return bundle
            local_retrieval_available = False

        for need in needs:
            retrieval_query = self.retrieval_query(
                need["question"],
                item_title=str(item["title"]) if need["scope"] == "item" else "",
            )
            if need["scope"] == "common":
                retrieval_query = (
                    f"{retrieval_query} {COMMON_KNOWLEDGE_RETRIEVAL_HINT}"
                ).strip()
            prepared: dict[str, object] | Mapping[str, object]
            if local_retrieval_available:
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
                    if need["scope"] == "common":
                        prepared = _common_rules_fallback(
                            need["question"],
                            failure_reason="knowledge_retrieval_failed",
                        )
                    elif not allow_web_fallback:
                        continue
                    else:
                        prepared = self._knowledge_bundle([], [], [], None)
                        prepared["issues"] = ["knowledge_retrieval_failed"]
            else:
                if need["scope"] == "common":
                    prepared = _common_rules_fallback(
                        need["question"],
                        failure_reason="knowledge_warmup_failed",
                    )
                elif not allow_web_fallback:
                    continue
                else:
                    prepared = self._knowledge_bundle([], [], [], None)
                    prepared["issues"] = ["knowledge_warmup_failed"]

            if need["scope"] == "item" and allow_web_fallback:
                prepared = self._knowledge_service.search_model_knowledge(
                    need["question"],
                    item=item,
                    local_prepared=prepared,
                    technical_service=self._technical_knowledge_service,
                )
            current_issues = prepared.get("issues")
            if isinstance(current_issues, list):
                for issue in current_issues:
                    if isinstance(issue, str) and issue not in issues:
                        issues.append(issue)

            prepared_results = prepared.get("results")
            if isinstance(prepared_results, list):
                results.extend(prepared_results)
            current_reliability = prepared.get("reliability")
            if reliability is None:
                reliability = current_reliability
            context = prepared.get("context")
            current_sources = valid_knowledge_sources(prepared)
            context_text = (
                str(context.get("context", "")).strip()
                if isinstance(context, Mapping)
                else ""
            )
            # ``can_answer`` is a reliability/debug signal from the old RAG
            # guard, not permission to discard retrieved evidence.  A context
            # can be useful even when it has no source metadata yet.
            if not context_text:
                if not (
                    isinstance(current_issues, list)
                    and any(isinstance(issue, str) for issue in current_issues)
                ):
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


def _common_rules_fallback(
    question: str,
    *,
    failure_reason: str,
) -> dict[str, object]:
    """Read checked-in common seller rules when the retrieval index is unavailable."""

    snippet = _common_rules_snippet(question)
    if not snippet:
        return {
            "can_answer": False,
            "context": None,
            "sources": [],
            "results": [],
            "reliability": {
                "can_answer": False,
                "next_step": "human_handoff",
                "reason": failure_reason,
                "top_rerank_score": None,
                "threshold": None,
            },
            "issues": [failure_reason, "common_rules_fallback_unavailable"],
        }
    sources = [dict(_COMMON_RULES_SOURCE)]
    return {
        "can_answer": True,
        "context": {"context": snippet, "sources": sources},
        "sources": sources,
        "results": [{"content": snippet, "source": _COMMON_RULES_FILE}],
        "reliability": {
            "can_answer": True,
            "next_step": "local_common_rules_fallback",
            "reason": "retrieval_index_unavailable",
            "top_rerank_score": None,
            "threshold": None,
        },
        "issues": [],
    }


def _common_rules_snippet(question: str) -> str:
    path = _common_rules_path()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        logger.exception("Local common seller rules unavailable at %s", path)
        return ""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        return ""

    keywords = _common_rules_keywords(question)
    specific_keywords = {
        keyword
        for keyword in keywords
        if keyword not in _COMMON_RULES_BASE_KEYWORDS
    }
    scored: list[tuple[int, int, str]] = []
    for index, paragraph in enumerate(paragraphs):
        score = sum(
            3 if keyword in specific_keywords else 1
            for keyword in keywords
            if keyword in paragraph
        )
        if score:
            scored.append((score, index, paragraph))
    if not scored:
        selected = paragraphs[:4]
    else:
        top_indexes = sorted(index for _score, index, _paragraph in sorted(
            scored,
            key=lambda item: (-item[0], item[1]),
        )[:8])
        selected = [paragraphs[index] for index in top_indexes]
    return _limit_text("\n\n".join(selected), 3200)


def _common_rules_keywords(question: str) -> tuple[str, ...]:
    query = str(question or "")
    keywords = list(_COMMON_RULES_BASE_KEYWORDS)
    if any(term in query for term in ("裂", "碎", "坏", "破损", "损坏", "质量问题", "有问题")):
        keywords.extend(_COMMON_RULES_DAMAGE_KEYWORDS)
    if any(term in query for term in ("运费", "邮费", "谁出", "承担")):
        keywords.extend(_COMMON_RULES_FEE_KEYWORDS)
    if any(term in query for term in ("七天", "七日", "无理由")):
        keywords.extend(_COMMON_RULES_SEVEN_DAY_KEYWORDS)
    if "闲鱼" in query:
        keywords.extend(_COMMON_RULES_XIANYU_KEYWORDS)
    return tuple(dict.fromkeys(keywords))


def _common_rules_path() -> Path:
    base = "data/xianyu/knowledge"
    if settings is not None:
        base = str(getattr(settings, "xianyu_knowledge_base_path", base) or base)
    return resolve_project_path(base) / "common" / _COMMON_RULES_FILE


def _limit_text(text: str, max_chars: int) -> str:
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rsplit("\n\n", 1)[0].strip() or cleaned[:max_chars].strip()
