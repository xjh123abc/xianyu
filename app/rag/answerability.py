"""Gate reranked evidence before it is passed to context construction."""

from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import Any, Literal, TypedDict

from config.settings import settings


NextStep = Literal["context_builder", "clarify"]
ReliabilityReason = Literal[
    "sufficient_evidence",
    "no_results",
    "no_valid_content",
    "missing_rerank_score",
    "score_below_threshold",
]


class ReliabilityResult(TypedDict):
    """Explicit decision consumed by the next pipeline stage."""

    can_answer: bool
    next_step: NextStep
    reason: ReliabilityReason
    top_rerank_score: float | None
    threshold: float


class AnswerReliability:
    """Decide whether reranked evidence is sufficient for downstream RAG."""

    def __init__(self, threshold: float | None = None) -> None:
        """Use an injected threshold or read the configured project threshold."""
        if threshold is None:
            if settings is None:
                raise RuntimeError("Project settings are unavailable")
            configured_model_id = str(
                getattr(settings, "answer_reliability_model_id", "")
            ).strip()
            configured_model_path = str(getattr(settings, "reranker_model_path", "")).strip()
            if (
                configured_model_id
                and configured_model_path
                and Path(configured_model_path).name != configured_model_id
            ):
                raise RuntimeError(
                    "The answer reliability threshold is calibrated for "
                    f"{configured_model_id}, not {Path(configured_model_path).name}"
                )
            threshold = getattr(settings, "answer_reliability_threshold", None)

        try:
            configured_threshold = float(threshold)
        except (TypeError, ValueError) as error:
            raise ValueError("answer_reliability_threshold must be a number") from error

        if not isfinite(configured_threshold):
            raise ValueError("answer_reliability_threshold must be finite")

        self.threshold = configured_threshold

    def evaluate(
        self,
        rerank_results: Sequence[Mapping[str, Any]] | None,
    ) -> ReliabilityResult:
        """Evaluate the first valid rerank result without re-sorting candidates.

        ``Reranker`` already returns results ordered by ``rerank_score``. This
        method therefore scans in the given order and uses the first candidate
        with non-empty content and a finite score as top 1.
        """
        if not rerank_results:
            return self._insufficient("no_results")

        has_valid_content = False
        has_missing_score = False

        for result in rerank_results:
            if not isinstance(result, Mapping):
                continue

            content = result.get("content")
            if not isinstance(content, str) or not content.strip():
                continue

            has_valid_content = True
            raw_score = result.get("rerank_score")
            if raw_score is None:
                has_missing_score = True
                continue

            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                has_missing_score = True
                continue

            if not isfinite(score):
                has_missing_score = True
                continue

            if score >= self.threshold:
                return {
                    "can_answer": True,
                    "next_step": "context_builder",
                    "reason": "sufficient_evidence",
                    "top_rerank_score": score,
                    "threshold": self.threshold,
                }

            return {
                "can_answer": False,
                "next_step": "clarify",
                "reason": "score_below_threshold",
                "top_rerank_score": score,
                "threshold": self.threshold,
            }

        if has_valid_content and has_missing_score:
            return self._insufficient("missing_rerank_score")
        return self._insufficient("no_valid_content")

    def _insufficient(self, reason: ReliabilityReason) -> ReliabilityResult:
        return {
            "can_answer": False,
            "next_step": "clarify",
            "reason": reason,
            "top_rerank_score": None,
            "threshold": self.threshold,
        }
