"""Evidence-bound answers for one item's facts and model knowledge."""

from __future__ import annotations

import re
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from app.generation.deepseek import DeepSeekGenerator
from app.services.intent_router import IntentMatch
from app.services.xianyu.experts.contracts import (
    ExpertContext,
    ExpertResult,
    ExpertTask,
)
from app.services.xianyu.item_fact_responder import ItemFactResponder
from app.services.xianyu.responses import requires_human_handoff, valid_knowledge_sources


EvidencePreparer = Callable[..., Awaitable[Mapping[str, object]]]
logger = logging.getLogger(__name__)


class ProductAgent:
    """Handle product facts or selected-item model knowledge, never delivery."""

    def __init__(
        self,
        *,
        fact_responder: ItemFactResponder,
        route_intent: Callable[[str], IntentMatch],
        prepare_evidence: EvidencePreparer,
        generator: Callable[[], DeepSeekGenerator],
    ) -> None:
        self._fact_responder = fact_responder
        self._route_intent = route_intent
        self._prepare_evidence = prepare_evidence
        self._generator = generator

    async def run(
        self,
        tasks: list[ExpertTask],
        context: ExpertContext,
    ) -> list[ExpertResult]:
        """Return exactly one result per supplied product task."""

        return [await self._run_one(task, context) for task in tasks]

    async def _run_one(self, task: ExpertTask, context: ExpertContext) -> ExpertResult:
        if task.expert != "product":
            return ExpertResult.handoff(task, "task_expert_mismatch")
        if context.item is None:
            return ExpertResult.handoff(task, "item_context_unavailable", missing_fields=("item",))
        if task.knowledge_scope == "item_fact":
            return self._answer_item_fact(task, context.item)
        if task.knowledge_scope != "model_knowledge":
            return ExpertResult.handoff(task, "unsupported_product_knowledge_scope")

        prepared = await self._prepare_evidence(
            task.normalized_question,
            item=context.item,
            scope="item",
        )
        return await self._answer_from_evidence(task, context, prepared)

    def _answer_item_fact(
        self,
        task: ExpertTask,
        item: Mapping[str, object],
    ) -> ExpertResult:
        response = self._fact_responder.answer_intent(
            task.normalized_question,
            item,
            self._route_intent(task.normalized_question),
        )
        sources = _sources(response)
        if response.get("action") == "reply":
            answer = response.get("answer")
            if isinstance(answer, str) and answer.strip():
                return ExpertResult.answered(task, answer.strip(), sources=sources)

        listing_answer = self._listing_description_answer(task.normalized_question, item)
        if listing_answer is not None:
            return ExpertResult.answered(task, listing_answer, sources=sources)
        return ExpertResult.handoff(
            task,
            str(response.get("reason") or "item_fact_unavailable"),
            missing_fields=tuple(self._route_intent(task.normalized_question).required_fields),
            sources=sources,
        )

    async def _answer_from_evidence(
        self,
        task: ExpertTask,
        context: ExpertContext,
        prepared: Mapping[str, object],
    ) -> ExpertResult:
        sources = valid_knowledge_sources(prepared)
        evidence = _context_text(prepared)
        issues = _issues(prepared)
        if not prepared.get("can_answer") or not evidence or not sources:
            return ExpertResult.handoff(
                task,
                issues[0] if issues else "model_knowledge_unavailable",
                missing_fields=("model_knowledge",),
                sources=sources,
            )
        try:
            answer = self._generator().generate_xianyu_expert(
                "product",
                task.normalized_question,
                context.item,
                evidence,
                history=context.history,
            )
        except Exception:
            logger.exception("Product expert generation failed for task_id=%s", task.task_id)
            return ExpertResult.handoff(task, "product_generation_failed", sources=sources)
        if not isinstance(answer, str) or not answer.strip():
            return ExpertResult.handoff(task, "product_generation_empty", sources=sources)
        if requires_human_handoff(answer):
            return ExpertResult.handoff(task, "generated_reply_requires_human_review", sources=sources)
        return ExpertResult.answered(task, answer.strip(), sources=sources)

    @staticmethod
    def _listing_description_answer(
        question: str,
        item: Mapping[str, object],
    ) -> str | None:
        """Return an exact seller-description sentence, never an inference."""

        facts = ItemFactResponder.structured_facts(item)
        conflicts = ItemFactResponder.fact_conflicts(facts)
        if conflicts & {"condition", "function", "history", "lens"}:
            return None
        description = facts.get("listing_description")
        if not isinstance(description, str) or not description.strip():
            return None
        keywords = tuple(
            term for term in ("快门", "过片", "光圈", "镜头", "成像", "漏光", "霉", "雾")
            if term in question.casefold()
        )
        if not keywords:
            return None
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[。！？!?])", description)
            if sentence.strip()
        ]
        for sentence in sentences:
            if any(keyword in sentence.casefold() for keyword in keywords):
                return sentence if sentence.endswith(("。", "！", "？", "!", "?")) else sentence + "。"
        return None


def _context_text(prepared: Mapping[str, object]) -> str:
    context = prepared.get("context")
    return str(context.get("context", "")).strip() if isinstance(context, Mapping) else ""


def _issues(prepared: Mapping[str, object]) -> tuple[str, ...]:
    raw = prepared.get("issues")
    return tuple(issue for issue in raw if isinstance(issue, str)) if isinstance(raw, list) else ()


def _sources(response: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    raw = response.get("sources")
    return tuple(source for source in raw if isinstance(source, Mapping)) if isinstance(raw, list) else ()
