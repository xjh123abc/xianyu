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

        answered = [result for result in results if result.status == "answered"]
        if len(answered) != len(results) or not answered:
            reason = next(
                (result.reason for result in results if result.reason),
                "task_answer_unavailable",
            )
            response = non_rag_response(
                query,
                "稍等我看看",
                can_answer=False,
                route="xianyu",
                action="handoff",
            )
            response["reason"] = reason
            response["next_step"] = "human_handoff"
            return response

        sources: list[dict[str, object]] = []
        for result in answered:
            for source in result.sources:
                if source not in sources:
                    sources.append(source)
        return {
            "query": query,
            "route": "unified",
            "action": "reply",
            "answer": join_answers([result.answer for result in answered]),
            "sources": sources,
            "results": [],
            "reliability": None,
            "next_step": None,
            "can_answer": True,
            "task_types": [task.task_type for task in tasks],
        }
