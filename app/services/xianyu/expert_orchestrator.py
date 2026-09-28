"""Single decision boundary for the Xianyu expert answer path."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from app.generation.deepseek import DeepSeekGenerator
from app.services.xianyu.experts.contracts import ExpertContext, ExpertResult, ExpertTask
from app.services.xianyu.experts.price_agent import PriceAgent
from app.services.xianyu.experts.product_agent import ProductAgent
from app.services.xianyu.experts.service_agent import ServiceAgent
from app.services.xianyu.item_fact_responder import ItemFactResponder
from app.services.xianyu.knowledge_responder import XianyuKnowledgeResponder
from config.settings import settings


logger = logging.getLogger(__name__)

ExpertProvider = Callable[[], DeepSeekGenerator] | DeepSeekGenerator


class XianyuExpertOrchestrator:
    """Plan, execute, validate, and merge one Xianyu buyer turn.

    This class is deliberately transport-free.  It returns the existing
    response dictionary and leaves message sending, seller notification, and
    human takeover to the S3 channel worker.
    """

    def __init__(
        self,
        *,
        fact_responder: ItemFactResponder,
        knowledge_responder: XianyuKnowledgeResponder,
        generator: ExpertProvider,
        product_agent: ProductAgent | None = None,
        price_agent: PriceAgent | None = None,
        service_agent: ServiceAgent | None = None,
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

        self._generator_provider = generator
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
                    if _allows_empty_answer(task):
                        completed[task.task_id] = result
                    else:
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

    def _get_generator(self) -> DeepSeekGenerator:
        provider = self._generator_provider
        if hasattr(provider, "generate_xianyu_expert"):
            return provider  # type: ignore[return-value]
        return provider()  # type: ignore[operator]

    @staticmethod
    def _remaining(deadline: float | None) -> float:
        if deadline is None:
            return 30.0
        return max(deadline - time.monotonic(), 0.001)


def _allows_empty_answer(task: ExpertTask) -> bool:
    return (
        task.knowledge_scope == "no_reply"
        or task.query_target == "no_reply"
        or task.intent_context.get("intent") == "service.no_reply"
    )
