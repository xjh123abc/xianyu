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
    _THANKS = "不客气，有需要随时说。"

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
            if task.query_target == "thanks":
                return ExpertResult.answered(task, self._THANKS)
            return ExpertResult.answered(task, self._GREETING)
        if task.knowledge_scope == "no_reply":
            return ExpertResult.answered(task, "")
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
        if (
            response.get("action") == "reply"
            and not response.get("reason")
            and isinstance(answer, str)
            and answer.strip()
        ):
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
        # ``can_answer`` and source validation are diagnostics only.  Generate
        # from every non-empty retrieval/rerank context during testing.
        if not evidence:
            return ExpertResult.handoff(
                task,
                issues[0] if issues else "seller_rule_unavailable",
                missing_fields=("seller_rule",),
                sources=sources,
            )
        evidence = _manual_review_evidence(task, context, evidence)
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
        safe_answer = _manual_review_safe_answer(task, context, answer)
        return ExpertResult.answered(
            task,
            safe_answer,
            sources=sources,
            evidence=evidence,
            raw_answer=answer,
        )


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


def _manual_review_evidence(
    task: ExpertTask,
    context: ExpertContext,
    evidence: str,
) -> str:
    """Add item-specific manual review limits without changing retrieved facts."""

    if not _requires_manual_review_guard(task, context):
        return evidence
    constraint = (
        "本轮售后处理约束：当前商品 after_sale.return_policy=manual_review。"
        "买家关于损坏、开裂、到货异常的说法只能视为待核实反馈或假设，"
        "不能自动认定为已确认商品事实或卖家责任。未核实、未获平台/卖家批准前，"
        "不得承诺同意退货退款、仅退款、换货、赔付或由卖家/我承担退回运费。"
        "只能说明需保留包装、面单、商品现状、照片/视频等证据，并在平台售后流程中"
        "按订单、商品资料、证据和责任归属核实；经核实属于卖家责任时，"
        "合理必要费用再按法律和平台规则处理。"
    )
    return f"{constraint}\n\n{evidence}"


def _manual_review_safe_answer(
    task: ExpertTask,
    context: ExpertContext,
    answer: str,
) -> str:
    """Keep manual-review after-sale replies conditional when the model overcommits."""

    if not _requires_manual_review_guard(task, context):
        return answer
    if not _has_unqualified_after_sale_commitment(answer):
        return answer
    intent = str(task.intent_context.get("intent") or "")
    if intent == "after_sale.return_shipping_fee":
        return (
            "退回运费要看退货原因和责任归属：如果只是买家个人原因退货，"
            "通常由买家承担；如果经核实属于商品质量问题、到货损坏或描述不符等卖家责任，"
            "合理必要退货费用再按平台规则和责任归属处理。你反馈的情况还需要先提交照片/视频、"
            "包装和面单等证据核实，我现在不能直接承诺由我承担。"
        )

    condition = _first_scenario(task)
    event = str(condition.get("event") or "商品问题")
    modality = condition.get("modality")
    if modality == "reported_unverified":
        return (
            f"你反馈的{event}我先按未核实售后反馈处理，不能直接认定责任或承诺退货退款。"
            "请先在闲鱼订单售后里提交开箱/包装/面单/商品现状的照片或视频，"
            "我会结合订单、商品资料、物流包装和平台规则核实。经核实属于到货损坏、"
            "商品质量问题或描述不符等卖家责任时，再按平台流程处理退货退款；"
            "合理必要退回费用也按平台规则和责任归属处理。"
        )
    if modality == "hypothetical":
        return (
            f"如果到手发现{event}，可以在平台售后里按商品问题/到货损坏提交证据申请处理。"
            "是否支持退货退款以及退回费用谁承担，需要看照片/视频、包装面单、订单情况和平台核实结果；"
            "未核实前我不能提前承诺一定同意退货退款或承担运费。"
        )
    return (
        f"关于{event}，当前需要先按售后流程核实，不能直接认定责任或承诺退货退款。"
        "请保留包装、面单、商品现状和照片/视频，在平台售后里提交；"
        "经核实属于卖家责任的，再按平台规则处理退货退款和合理必要费用。"
    )


def _requires_manual_review_guard(task: ExpertTask, context: ExpertContext) -> bool:
    intent = str(task.intent_context.get("intent") or "")
    if not intent.startswith("after_sale."):
        return False
    item = context.item
    if not isinstance(item, Mapping):
        return False
    facts = item.get("facts")
    after_sale = facts.get("after_sale") if isinstance(facts, Mapping) else None
    return isinstance(after_sale, Mapping) and after_sale.get("return_policy") == "manual_review"


def _has_unqualified_after_sale_commitment(answer: str) -> bool:
    normalized = str(answer or "")
    commitment_terms = (
        "会同意退货",
        "同意退货",
        "同意退款",
        "同意退货退款",
        "可以直接退款",
        "直接退款",
        "运费由我承担",
        "我承担运费",
        "卖家承担运费",
        "由卖家承担",
        "由我承担",
        "不用你承担",
        "不应由你承担",
    )
    if not any(term in normalized for term in commitment_terms):
        return False
    conditional_terms = (
        "经核实",
        "核实后",
        "确认属于",
        "责任归属",
        "平台规则",
        "平台核实",
        "以平台",
        "未核实",
        "不能直接",
    )
    return not any(term in normalized for term in conditional_terms)


def _first_scenario(task: ExpertTask) -> Mapping[str, object]:
    conditions = task.intent_context.get("conditions")
    if isinstance(conditions, list):
        for condition in conditions:
            if isinstance(condition, Mapping) and condition.get("type") == "scenario":
                return condition
    return {}
