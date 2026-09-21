"""Evidence-bound answers for seller service rules and greetings."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping

from app.generation.deepseek import DeepSeekGenerator
from app.services.xianyu.experts.contracts import ExpertContext, ExpertResult, ExpertTask
from app.services.xianyu.item_fact_responder import ItemFactResponder
from app.services.xianyu.responses import valid_knowledge_sources


EvidencePreparer = Callable[..., Awaitable[Mapping[str, object]]]
logger = logging.getLogger(__name__)


class ServiceAgent:
    """Handle shipping, after-sale, common seller rules, and simple greetings."""

    _GREETING = "你好，有什么想了解的？"

    def __init__(
        self,
        *,
        fact_responder: ItemFactResponder,
        prepare_evidence: EvidencePreparer,
        generator: Callable[[], DeepSeekGenerator],
    ) -> None:
        self._fact_responder = fact_responder
        self._prepare_evidence = prepare_evidence
        self._generator = generator

    async def run(self, tasks: list[ExpertTask], context: ExpertContext) -> list[ExpertResult]:
        """Return exactly one result per supplied service task."""

        return [await self._run_one(task, context) for task in tasks]

    async def _run_one(self, task: ExpertTask, context: ExpertContext) -> ExpertResult:
        if task.expert != "service":
            return ExpertResult.handoff(task, "task_expert_mismatch")
        if task.knowledge_scope == "greeting":
            return ExpertResult.answered(task, self._GREETING)
        if task.knowledge_scope == "item_fact":
            return self._answer_item_fact(task, context)
        if task.knowledge_scope != "seller_rule":
            return ExpertResult.handoff(task, "unsupported_service_knowledge_scope")

        prepared = await self._prepare_evidence(
            task.normalized_question,
            item=context.item,
            scope="common",
        )
        return await self._answer_from_evidence(task, context, prepared)

    def _answer_item_fact(self, task: ExpertTask, context: ExpertContext) -> ExpertResult:
        if context.item is None:
            return ExpertResult.handoff(task, "item_context_unavailable", missing_fields=("item",))
        response = self._fact_responder.answer_target(
            task.original_question,
            context.item,
            task.query_target,
        )
        sources = _sources(response)
        answer = response.get("answer")
        if response.get("action") == "reply" and isinstance(answer, str) and answer.strip():
            return ExpertResult.answered(task, answer.strip(), sources=sources)
        return ExpertResult.handoff(
            task,
            str(response.get("reason") or "seller_service_fact_unavailable"),
            missing_fields=self._fact_responder.required_fields_for_target(
                task.query_target
            ),
            sources=sources,
        )

    async def _answer_from_evidence(
        self, task: ExpertTask, context: ExpertContext, prepared: Mapping[str, object]
    ) -> ExpertResult:
        sources = valid_knowledge_sources(prepared)
        evidence = _context_text(prepared)
        issues = _issues(prepared)
        if not prepared.get("can_answer") or not evidence or not sources:
            return ExpertResult.handoff(
                task,
                issues[0] if issues else "seller_rule_unavailable",
                missing_fields=("seller_rule",),
                sources=sources,
            )
        timeout_seconds = _remaining_timeout(context)
        if timeout_seconds is not None and timeout_seconds <= 0:
            return ExpertResult.handoff(task, "expert_processing_timeout", sources=sources)
        try:
            generator = self._generator()
            kwargs: dict[str, object] = {"history": context.history}
            if timeout_seconds is not None:
                kwargs["timeout_seconds"] = timeout_seconds
            answer = await asyncio.to_thread(
                generator.generate_xianyu_expert,
                "service",
                task.original_question,
                context.item,
                evidence,
                **kwargs,
            )
        except Exception:
            logger.exception("Service expert generation failed for task_id=%s", task.task_id)
            return ExpertResult.handoff(task, "service_generation_failed", sources=sources)
        if not isinstance(answer, str) or not answer.strip():
            return ExpertResult.handoff(task, "service_generation_empty", sources=sources)
        return ExpertResult.answered(task, answer.strip(), sources=sources)


def _context_text(prepared: Mapping[str, object]) -> str:
    context = prepared.get("context")
    return str(context.get("context", "")).strip() if isinstance(context, Mapping) else ""


def _issues(prepared: Mapping[str, object]) -> tuple[str, ...]:
    raw = prepared.get("issues")
    return tuple(issue for issue in raw if isinstance(issue, str)) if isinstance(raw, list) else ()


def _sources(response: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    raw = response.get("sources")
    return tuple(source for source in raw if isinstance(source, Mapping)) if isinstance(raw, list) else ()


def _remaining_timeout(context: ExpertContext) -> float | None:
    if context.deadline is None:
        return None
    return context.deadline - time.monotonic()
