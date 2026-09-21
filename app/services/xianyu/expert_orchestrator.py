"""Single decision boundary for the Xianyu expert answer path."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from app.generation.deepseek import DeepSeekGenerator
from app.services.intent_router import IntentRouter
from app.services.query_planner import build_expert_plan
from app.services.xianyu.experts.contracts import ExpertContext, ExpertResult, ExpertTask
from app.services.xianyu.experts.price_agent import PriceAgent
from app.services.xianyu.experts.product_agent import ProductAgent
from app.services.xianyu.experts.service_agent import ServiceAgent
from app.services.xianyu.item_fact_responder import ItemFactResponder
from app.services.xianyu.knowledge_responder import XianyuKnowledgeResponder
from app.services.xianyu.responses import (
    clarification,
    common_handoff,
    handoff,
    item_source,
)
from config.settings import settings


logger = logging.getLogger(__name__)

ExpertProvider = Callable[[], DeepSeekGenerator] | DeepSeekGenerator


class XianyuExpertOrchestrator:
    """Plan, execute, validate, and merge one Xianyu buyer turn.

    This class is deliberately transport-free.  It returns the existing
    response dictionary and leaves message sending, seller notification, and
    human takeover to the S3 channel worker.
    """

    _PLANNER_BUDGET_SECONDS = 8.0

    def __init__(
        self,
        *,
        fact_responder: ItemFactResponder,
        knowledge_responder: XianyuKnowledgeResponder,
        intent_router: IntentRouter,
        generator: ExpertProvider,
        product_agent: ProductAgent | None = None,
        price_agent: PriceAgent | None = None,
        service_agent: ServiceAgent | None = None,
        planner: Callable[..., object] | None = None,
        budget_seconds: float | None = None,
    ) -> None:
        configured_budget = getattr(settings, "xianyu_expert_budget_seconds", 25.0)
        requested_budget = float(
            configured_budget if budget_seconds is None else budget_seconds
        )
        channel_timeout = float(
            getattr(settings, "xianyu_chat_api_timeout_seconds", 30.0)
        )
        if requested_budget <= 0 or channel_timeout <= 0:
            raise ValueError("expert and channel budgets must be greater than zero")
        # Leave a transport margin so a completed expert decision can still be
        # serialized and handed to the channel before its deadline.
        self.budget_seconds = min(requested_budget, channel_timeout * 0.9)

        self._intent_router = intent_router
        self._generator_provider = generator
        self._planner = planner
        self._agents: dict[str, Any] = {}

        self._agents["product"] = product_agent or ProductAgent(
            fact_responder=fact_responder,
            prepare_evidence=knowledge_responder.prepare_evidence,
            generator=self._get_generator,
        )
        self._agents["price"] = price_agent or PriceAgent()
        self._agents["service"] = service_agent or ServiceAgent(
            fact_responder=fact_responder,
            prepare_evidence=knowledge_responder.prepare_evidence,
            generator=self._get_generator,
        )

    async def handle(
        self,
        query: str,
        *,
        item: Mapping[str, object] | None,
        history: Sequence[Mapping[str, object]] | None = None,
        session_state: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Return exactly one reply or one fixed human-handoff decision."""

        normalized_query = str(query or "").strip()
        if not normalized_query:
            return self._handoff_response(
                normalized_query,
                item,
                (),
                "empty_xianyu_query",
            )

        deadline = time.monotonic() + self.budget_seconds
        xianyu_context = self._xianyu_context(session_state)
        context = ExpertContext(
            query=normalized_query,
            item=item,
            history=tuple(history or ()),
            xianyu_context=xianyu_context,
            deadline=deadline,
        )

        try:
            tasks = await asyncio.wait_for(
                asyncio.to_thread(self.build_plan, context),
                timeout=self._remaining(deadline),
            )
        except asyncio.TimeoutError:
            return self._handoff_response(
                normalized_query,
                item,
                (),
                "expert_planning_timeout",
            )
        except Exception:
            logger.exception("Xianyu expert planning failed")
            return self._handoff_response(
                normalized_query,
                item,
                (),
                "expert_planning_failed",
            )

        if not tasks:
            return self._handoff_response(
                normalized_query,
                item,
                (),
                "expert_plan_empty",
            )

        results = await self.execute_tasks(tasks, context)
        return self._merge(normalized_query, item, tasks, results)

    def build_plan(self, context: ExpertContext) -> list[ExpertTask]:
        """Compatibility planner used only by the legacy ``handle`` entrypoint."""

        planner = self._planner or self._model_planner(context.deadline)
        planning_context = dict(context.xianyu_context)
        if context.item is not None and not planning_context.get("item_id"):
            planning_context["item_id"] = context.item.get("item_id")
        tasks = build_expert_plan(
            context.query,
            intent_router=self._intent_router,
            planner=planner,
            history=context.history,
            xianyu_context=planning_context,
        )
        if not tasks and context.item is not None:
            # An item-scoped question that misses the deterministic vocabulary
            # still needs a product task.  The product expert will gate it on
            # scoped evidence; an empty plan would silently skip the evidence
            # check and lose the S3 A09 fail-closed decision.
            return [
                ExpertTask(
                    "fallback-product",
                    "product",
                    context.query,
                    "商品专项知识",
                    "model_knowledge",
                    query_target="product.model_knowledge",
                )
            ]
        return tasks

    def _build_plan(self, context: ExpertContext) -> list[ExpertTask]:
        """Deprecated private alias retained for callers outside the main chain."""

        return self.build_plan(context)

    async def execute_tasks(
        self,
        tasks: Sequence[ExpertTask],
        context: ExpertContext,
    ) -> list[ExpertResult]:
        """Execute already-planned expert tasks without planning them again."""

        if not tasks:
            return []
        deadline = context.deadline or (time.monotonic() + self.budget_seconds)
        execution_context = (
            context if context.deadline is not None else replace(context, deadline=deadline)
        )
        return await self._execute(tasks, execution_context, deadline)

    def _model_planner(self, deadline: float | None) -> Callable[..., object] | None:
        generator = self._get_generator()
        method = getattr(generator, "plan_xianyu_questions", None)
        if not callable(method):
            return None

        def plan(query: str, *, history: Sequence[Mapping[str, object]] | None = None) -> object:
            kwargs: dict[str, object] = {"history": history}
            timeout = min(
                self._remaining(deadline),
                self._PLANNER_BUDGET_SECONDS,
            )
            if timeout is not None and _accepts_keyword(method, "timeout_seconds"):
                kwargs["timeout_seconds"] = timeout
            return method(query, **kwargs)

        return plan

    async def _execute(
        self,
        tasks: Sequence[ExpertTask],
        context: ExpertContext,
        deadline: float,
    ) -> list[ExpertResult]:
        pending = {task.task_id: task for task in tasks}
        completed: dict[str, ExpertResult] = {}

        while pending:
            ready = [
                task
                for task in pending.values()
                if all(dependency in completed for dependency in task.depends_on_task_ids)
            ]
            if not ready:
                for task in pending.values():
                    completed[task.task_id] = ExpertResult.handoff(
                        task,
                        "dependency_graph_invalid",
                    )
                break

            runnable: list[ExpertTask] = []
            for task in ready:
                failed_dependencies = [
                    dependency
                    for dependency in task.depends_on_task_ids
                    if completed[dependency].status != "answered"
                ]
                if failed_dependencies:
                    completed[task.task_id] = ExpertResult.handoff(
                        task,
                        "dependency_unresolved:" + ",".join(failed_dependencies),
                    )
                else:
                    runnable.append(task)
                pending.pop(task.task_id, None)

            if not runnable:
                continue
            if self._remaining(deadline) <= 0:
                for task in runnable:
                    completed[task.task_id] = ExpertResult.handoff(
                        task,
                        "expert_processing_timeout",
                    )
                continue

            batches = {
                expert: [task for task in runnable if task.expert == expert]
                for expert in {task.expert for task in runnable}
            }
            batch_coroutines = [
                self._run_batch(expert, batch, context)
                for expert, batch in batches.items()
            ]
            try:
                batch_results = await asyncio.wait_for(
                    asyncio.gather(*batch_coroutines, return_exceptions=True),
                    timeout=self._remaining(deadline),
                )
            except asyncio.TimeoutError:
                for task in runnable:
                    completed[task.task_id] = ExpertResult.handoff(
                        task,
                        "expert_processing_timeout",
                    )
                continue

            for (expert, batch), outcome in zip(batches.items(), batch_results):
                if isinstance(outcome, BaseException):
                    logger.warning(
                        "Xianyu %s expert batch failed",
                        expert,
                        exc_info=(type(outcome), outcome, outcome.__traceback__),
                    )
                    for task in batch:
                        completed[task.task_id] = ExpertResult.handoff(
                            task,
                            "expert_execution_failed",
                        )
                    continue
                self._record_batch_results(
                    batch,
                    outcome,
                    completed,
                )

        return [completed[task.task_id] for task in tasks]

    async def _run_batch(
        self,
        expert: str,
        tasks: list[ExpertTask],
        context: ExpertContext,
    ) -> object:
        runner = self._agents.get(expert)
        if runner is None or not callable(getattr(runner, "run", None)):
            return [ExpertResult.handoff(task, "expert_runner_unavailable") for task in tasks]
        result = runner.run(tasks, context)
        return await result if inspect.isawaitable(result) else result

    @staticmethod
    def _record_batch_results(
        tasks: Sequence[ExpertTask],
        raw_results: object,
        completed: dict[str, ExpertResult],
    ) -> None:
        by_id: dict[str, ExpertResult] = {}
        if isinstance(raw_results, list):
            for result in raw_results:
                if isinstance(result, ExpertResult) and result.task_id not in by_id:
                    by_id[result.task_id] = result
        for task in tasks:
            result = by_id.get(task.task_id)
            if result is None or result.expert != task.expert:
                completed[task.task_id] = ExpertResult.handoff(
                    task,
                    "expert_result_contract_invalid",
                )
                continue
            if result.status == "answered":
                if not isinstance(result.answer, str) or not result.answer.strip():
                    completed[task.task_id] = ExpertResult.handoff(
                        task,
                        "expert_result_empty",
                        sources=result.sources,
                    )
                else:
                    completed[task.task_id] = result
            elif result.status == "handoff":
                completed[task.task_id] = result
            else:
                completed[task.task_id] = ExpertResult.handoff(
                    task,
                    "expert_result_status_invalid",
                )

    def _merge(
        self,
        query: str,
        item: Mapping[str, object] | None,
        tasks: Sequence[ExpertTask],
        results: Sequence[ExpertResult],
    ) -> dict[str, object]:
        answers: list[str] = []
        sources: list[Mapping[str, object]] = []
        seen_answers: set[str] = set()
        for result in results:
            answer = (result.answer or "").strip()
            normalized = re_space(answer)
            if answer and normalized not in seen_answers:
                answers.append(answer)
                seen_answers.add(normalized)
            for source in valid_result_sources(result):
                if source not in sources:
                    sources.append(source)

        if item is not None:
            item_evidence = item_source(item)
            if item_evidence not in sources:
                sources.insert(0, item_evidence)
        failures = [result for result in results if result.status != "answered"]
        unavailable = [
            f"{task.question_fragment}暂时无法确认。"
            for task, result in zip(tasks, results)
            if result.status != "answered"
        ]
        if not answers:
            # A planner/expert failure is not something the buyer can resolve
            # by repeating the question.  Preserve AUTO mode and return the
            # standard unavailable result instead of an old clarification.
            return self._handoff_response(
                query,
                item,
                results,
                self._handoff_reason(tasks, results),
            )
        response: dict[str, object] = {
            "query": query,
            "route": "xianyu",
            "action": "reply",
            "answer": "\n".join([*answers, *unavailable]),
            "sources": sources,
            "results": [],
            "reliability": None,
            "next_step": None,
            "can_answer": not failures,
        }
        if item is not None:
            response.update({"item_id": item["item_id"], "item_info": dict(item)})
        return response

    def _handoff_response(
        self,
        query: str,
        item: Mapping[str, object] | None,
        results: Sequence[ExpertResult],
        reason: str,
    ) -> dict[str, object]:
        response = handoff(query, reason, item) if item is not None else common_handoff(query, reason)
        sources: list[Mapping[str, object]] = []
        if item is not None:
            sources.append(item_source(item))
        for result in results:
            for source in valid_result_sources(result):
                if source not in sources:
                    sources.append(source)
        response["sources"] = sources
        response["results"] = []
        response["reliability"] = None
        return response

    @staticmethod
    def _handoff_reason(
        tasks: Sequence[ExpertTask],
        results: Sequence[ExpertResult],
    ) -> str:
        task_by_id = {task.task_id: task for task in tasks}
        failures = [result for result in results if result.status != "answered"]
        confirmed = [
            result.answer.strip()
            for result in results
            if result.status == "answered"
            and isinstance(result.answer, str)
            and result.answer.strip()
        ]
        if len(failures) == 1 and not confirmed:
            return failures[0].reason or "expert_answer_unavailable"
        missing = []
        for result in failures:
            task = task_by_id.get(result.task_id)
            label = task.question_fragment if task else result.task_id
            detail = result.reason or "expert_answer_unavailable"
            missing.append(f"{label}:{detail}")
        reason = "缺少或无法确认：" + "；".join(missing)
        if confirmed:
            reason += "；已确认：" + "；".join(confirmed)
        return reason

    def _get_generator(self) -> DeepSeekGenerator:
        provider = self._generator_provider
        if hasattr(provider, "generate_xianyu_expert"):
            return provider  # type: ignore[return-value]
        return provider()  # type: ignore[operator]

    @staticmethod
    def _xianyu_context(session_state: Mapping[str, object] | None) -> Mapping[str, object]:
        if not isinstance(session_state, Mapping):
            return {}
        nested = session_state.get("xianyu_context")
        if isinstance(nested, Mapping):
            return dict(nested)
        if any(key in session_state for key in ("recent_price_topic", "shipping_condition")):
            return dict(session_state)
        return {}

    @staticmethod
    def _remaining(deadline: float | None) -> float:
        if deadline is None:
            return 30.0
        return max(deadline - time.monotonic(), 0.001)


def _accepts_keyword(method: Callable[..., object], name: str) -> bool:
    try:
        parameters = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == name or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def valid_result_sources(result: ExpertResult) -> tuple[Mapping[str, object], ...]:
    return tuple(
        source
        for source in result.sources
        if isinstance(source, Mapping)
        and isinstance(source.get("source"), str)
        and bool(source.get("source", "").strip())
        and source.get("index") is not None
    )


def re_space(value: str) -> str:
    return " ".join(value.split())
