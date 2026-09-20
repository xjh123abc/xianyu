"""Scoped Xianyu knowledge retrieval and grounded answer generation."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from app.generation.deepseek import DeepSeekGenerator
from app.services.intent_router import IntentMatch
from app.services.query_planner import (
    COMMON_KNOWLEDGE_RETRIEVAL_HINT,
    QuestionPlan,
    build_question_plan,
)
from app.services.knowledge_service import KnowledgeService
from app.services.xianyu.item_fact_responder import ItemFactResponder
from app.services.xianyu.responses import (
    clarification,
    common_handoff,
    handoff,
    join_answers,
    reply,
    requires_human_handoff,
    unavailable,
    valid_knowledge_sources,
)


logger = logging.getLogger(__name__)


_FACT_QUERY_PATTERNS = (
    re.compile(r"价格(?:是|为)?多少(?:元)?", re.IGNORECASE),
    re.compile(r"(?:多少钱|标价|售价|多少元|price|cost)", re.IGNORECASE),
    re.compile(
        r"(?:这个商品|商品)?(?:现在|当前)?(?:还)?(?:在吗|有吗|有货吗|在售吗|还有吗)",
        re.IGNORECASE,
    ),
    re.compile(r"(?:售卖)?状态(?:如何|怎么样|是什么)?", re.IGNORECASE),
    re.compile(r"(?:卖出|售出|已售|available|sold)", re.IGNORECASE),
)


class XianyuKnowledgeResponder:
    """Retrieve only the evidence scope required by an item/seller question."""

    def __init__(
        self,
        *,
        knowledge_service: KnowledgeService,
        generator: Callable[[], DeepSeekGenerator],
        fact_responder: ItemFactResponder,
        route_intent: Callable[[str], IntentMatch],
    ) -> None:
        self._knowledge_service = knowledge_service
        self._generator = generator
        self._fact_responder = fact_responder
        self._route_intent = route_intent

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
        """Answer item facts first, then use only scoped RAG for the remainder."""

        plan = plan or build_question_plan(query)
        if intent_match is not None and intent_match.intent == "BARGAIN":
            bargain_response = self._fact_responder.answer_intent(
                query,
                item,
                intent_match,
            )
            return bargain_response

        planned_facts = self._fact_responder.answer_plan(query, item, plan)
        if planned_facts.has_conflict:
            return handoff(
                query,
                "item_fact_conflict",
                item,
            )

        fact_answers = planned_facts.answers
        unresolved_answers = planned_facts.unresolved
        asks_knowledge = bool(plan["knowledge_questions"])
        if not asks_knowledge:
            if unresolved_answers:
                return handoff(query, join_answers(fact_answers + unresolved_answers), item)
            if force_full_item_answer:
                return self._generate_from_item_facts(
                    query,
                    item,
                    history=history,
                    fact_answers=fact_answers,
                )
            if fact_answers:
                return reply(query, join_answers(fact_answers), item)
            return clarification(query, item_id=str(item["item_id"]))

        if (
            not unresolved_answers
            and self._knowledge_questions_have_confirmed_facts(plan, item)
        ):
            return self._generate_from_item_facts(
                query,
                item,
                history=history,
                fact_answers=fact_answers,
            )

        prepared, knowledge_issues = await self._collect_knowledge(
            plan,
            item,
            remove_facts=bool(fact_answers),
        )
        unresolved_answers.extend(knowledge_issues)
        context = prepared.get("context")
        knowledge_sources = valid_knowledge_sources(prepared)
        if not isinstance(context, Mapping) or not knowledge_sources:
            return self._knowledge_unavailable(
                query,
                join_answers(fact_answers),
                item,
                prepared,
            )
        context_text = str(context.get("context", "")).strip()
        if not context_text:
            return self._knowledge_unavailable(
                query,
                join_answers(
                    fact_answers,
                ),
                item,
                prepared,
            )
        try:
            item_reference = ""
            if any(need["scope"] == "item" for need in plan["knowledge_questions"]):
                item_reference = (
                    f"当前商品：{item['title']}\n"
                    "未记录的信息不得推断为不存在。\n\n"
                )
            answer = self._generator().generate_xianyu(
                query,
                item,
                item_reference + "卖家规则与商品说明：\n" + context_text,
                history=history,
            )
            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError("empty Xianyu answer")
            answer = answer.strip()
            if unresolved_answers:
                answer = join_answers(fact_answers, answer)
            answer = join_answers([answer] + unresolved_answers)
        except Exception:
            logger.exception("Xianyu answer generation failed for item_id=%s", item["item_id"])
            return handoff(
                query,
                join_answers(
                    fact_answers + unresolved_answers,
                    "xianyu_generation_failed",
                ),
                item,
                prepared,
            )
        if unresolved_answers:
            return handoff(query, answer, item, prepared)
        return {
            "query": query,
            "route": "xianyu",
            "action": "reply",
            "answer": answer,
            "sources": [
                {"source": "mcp:get_item_info", "index": item["item_id"]},
                *knowledge_sources,
            ],
            "results": prepared.get("results", []),
            "reliability": prepared.get("reliability"),
            "next_step": None,
            "can_answer": True,
            "item_id": item["item_id"],
            "item_info": item,
        }

    def _knowledge_questions_have_confirmed_facts(
        self,
        plan: QuestionPlan,
        item: Mapping[str, object],
    ) -> bool:
        """Allow fact-only LLM composition when every remaining need is known."""

        return bool(plan["knowledge_questions"]) and all(
            self._fact_responder.answer_intent(
                need["question"],
                item,
                self._route_intent(need["question"]),
            ).get("action") == "reply"
            for need in plan["knowledge_questions"]
        )

    def _generate_from_item_facts(
        self,
        query: str,
        item: Mapping[str, object],
        *,
        history: Sequence[Mapping[str, Any]] | None,
        fact_answers: Sequence[str],
    ) -> dict[str, object]:
        """Use the full buyer message when confirmed facts need composing."""

        try:
            answer = self._generator().generate_xianyu(
                query,
                item,
                "当前商品的已确认事实已提供；没有额外卖家规则。"
                "请只依据这些事实回答完整买家问题。",
                history=history,
            )
            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError("empty fact-only Xianyu answer")
            return reply(query, answer.strip(), item)
        except Exception:
            logger.exception("Fact-only Xianyu answer generation failed for item_id=%s", item["item_id"])
            return handoff(
                query,
                join_answers(
                    fact_answers,
                    "fact_only_generation_failed",
                ),
                item,
            )

    async def handle_common(self, query: str) -> dict[str, object]:
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
            if requires_human_handoff(answer):
                return self._common_handoff(query, "generated_reply_requires_human_review")
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
            remove_facts=False,
        )
        return prepared

    async def _collect_knowledge(
        self,
        plan: QuestionPlan,
        item: Mapping[str, object],
        *,
        remove_facts: bool,
    ) -> tuple[dict[str, object], list[str]]:
        """Compatibility wrapper for the legacy whole-message responder."""

        prepared = await self._prepare_evidence_needs(
            plan["knowledge_questions"],
            item,
            remove_facts=remove_facts,
        )
        issues = prepared.get("issues")
        return (
            prepared,
            [issue for issue in issues if isinstance(issue, str)]
            if isinstance(issues, list)
            else [],
        )

    async def _prepare_evidence_needs(
        self,
        needs: Sequence[Mapping[str, str]],
        item: Mapping[str, object],
        *,
        remove_facts: bool,
    ) -> dict[str, object]:
        """Shared retrieval implementation for legacy and expert callers."""

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
                remove_facts=remove_facts,
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
        remove_facts: bool,
        item_title: str,
    ) -> str:
        """Keep structured fact wording from weakening knowledge retrieval."""

        cleaned = query
        if remove_facts:
            for pattern in _FACT_QUERY_PATTERNS:
                cleaned = pattern.sub(" ", cleaned)
        cleaned = re.sub(r"(?:这个|这件)?商品", " ", cleaned)
        cleaned = re.sub(r"[\s，,。.!！?？、；;：:]+", " ", cleaned).strip()
        cleaned = re.sub(r"^(?:和|及|与)\s*", "", cleaned).strip()
        normalized_title = item_title.strip()
        if normalized_title and normalized_title.casefold() not in cleaned.casefold():
            cleaned = f"{normalized_title} {cleaned}".strip()
        return cleaned or query

    @staticmethod
    def _common_handoff(query: str, answer: str) -> dict[str, object]:
        return common_handoff(query, answer)

    @staticmethod
    def _knowledge_unavailable(
        query: str,
        reason: str,
        item: Mapping[str, object],
        prepared: Mapping[str, object],
    ) -> dict[str, object]:
        """Expose unavailable evidence as a safe clarification, never handoff."""

        answer = join_answers(
            [reason],
            "暂无可确认的商品知识补充信息。",
        )
        response = unavailable(query, answer, item=item, prepared=prepared)
        response["reason"] = "knowledge_evidence_unavailable"
        return response
