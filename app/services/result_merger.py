"""Turn executor outcomes into the established chat response dictionary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from app.services.chat_contracts import Task, TaskResult
from app.services.chat_response import non_rag_response
from app.services.xianyu.responses import join_answers


class ResultMerger:
    """Merge ordered task results without introducing a channel-specific contract."""

    def merge(
        self,
        query: str,
        tasks: Sequence[Task],
        results: Sequence[TaskResult],
    ) -> dict[str, object]:
        """Keep a handler's legacy response intact when it is the only result."""

        raw_responses = [
            result.metadata.get("response")
            for result in results
            if isinstance(result.metadata.get("response"), Mapping)
        ]
        if len(results) == 1 and len(raw_responses) == 1:
            response = dict(raw_responses[0])
            response["query"] = query
            return response

        sources: list[dict[str, object]] = []
        task_by_id = {task.task_id: task for task in tasks}
        answer_parts: list[str] = []
        has_answered = False
        all_clarify = bool(results)
        for result in results:
            for source in result.sources:
                if source not in sources:
                    sources.append(source)
            task = task_by_id.get(result.task_id)
            answer = self._result_text(result, task)
            if answer and answer not in answer_parts:
                answer_parts.append(answer)
            has_answered = has_answered or result.status == "answered"
            all_clarify = all_clarify and result.status == "clarify"

        if not has_answered and all_clarify:
            response = non_rag_response(
                query,
                join_answers(answer_parts),
                can_answer=False,
                route="unified",
                action="clarify",
            )
            response["reason"] = next(
                (result.reason for result in results if result.reason),
                "task_answer_unavailable",
            )
            return response
        response = dict(raw_responses[0]) if raw_responses else {}
        response.update({
            "query": query,
            "route": "unified",
            "action": "reply",
            "answer": join_answers(answer_parts),
            "sources": sources,
            "results": [],
            "reliability": None,
            "next_step": None,
            "can_answer": all(result.status == "answered" for result in results),
            "task_types": [task.task_type for task in tasks],
            "evidence": _debug_values(raw_responses, "evidence"),
            "raw_answer": _debug_values(raw_responses, "raw_answer"),
        })
        if not response["can_answer"]:
            response["reason"] = next(
                (result.reason for result in results if result.reason),
                "task_answer_unavailable",
            )
        return response

    @staticmethod
    def _result_text(result: TaskResult, task: Task | None) -> str:
        """Render every executor outcome without leaking an internal reason."""

        del task
        return result.answer if isinstance(result.answer, str) else ""


def _debug_values(
    responses: Sequence[object], key: str
) -> str | list[str] | None:
    values = [
        value
        for response in responses
        if isinstance(response, Mapping)
        for value in [response.get(key)]
        if isinstance(value, str) and value
    ]
    if not values:
        return None
    return values[0] if len(values) == 1 else values
