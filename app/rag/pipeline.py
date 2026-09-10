"""RAG pipeline from reranked evidence to a grounded answer."""

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any, Literal, TypedDict

from app.generation.deepseek import DeepSeekGenerator
from app.rag.answerability import AnswerReliability, ReliabilityResult
from app.rag.context_builder import (
    ContextBuildResult,
    ContextBuilder,
    SourceReference,
)
from config.settings import settings


class PipelineResult(TypedDict):
    """Result returned after reliability and optional answer generation."""

    can_answer: bool
    next_step: Literal["llm", "complete", "human_handoff"]
    reliability: ReliabilityResult
    context: ContextBuildResult | None
    answer: str | None
    sources: list[SourceReference]


class RAGPipeline:
    """Run reliability, context construction, and grounded generation."""

    def __init__(
        self,
        answer_reliability: AnswerReliability | None = None,
        context_builder: ContextBuilder | None = None,
        generator: DeepSeekGenerator | None = None,
        *,
        context_max_chunks: int | None = None,
        context_score_gap: float | None = None,
    ) -> None:
        self.answer_reliability = answer_reliability or AnswerReliability()
        self.context_builder = context_builder or ContextBuilder()
        self.generator = generator or DeepSeekGenerator()
        self.context_max_chunks = int(
            context_max_chunks
            if context_max_chunks is not None
            else settings.rag_context_max_chunks
        )
        self.context_score_gap = float(
            context_score_gap
            if context_score_gap is not None
            else settings.rag_context_score_gap
        )
        if self.context_max_chunks <= 0:
            raise ValueError("context_max_chunks must be greater than zero")
        if not isfinite(self.context_score_gap) or self.context_score_gap < 0:
            raise ValueError("context_score_gap must be a finite non-negative number")

    def run_after_rerank(
        self,
        rerank_results: Sequence[Mapping[str, Any]] | None,
    ) -> PipelineResult:
        """Gate reranked evidence, then build context only when allowed."""
        reliability = self.answer_reliability.evaluate(rerank_results)
        if not reliability["can_answer"]:
            return {
                "can_answer": False,
                "next_step": "human_handoff",
                "reliability": reliability,
                "context": None,
                "answer": None,
                "sources": [],
            }

        selected_results = self._select_context_evidence(
            rerank_results,
            reliability["top_rerank_score"],
        )
        context = self.context_builder.build(selected_results)
        if not context["context"]:
            return {
                "can_answer": False,
                "next_step": "human_handoff",
                "reliability": reliability,
                "context": None,
                "answer": None,
                "sources": [],
            }
        return {
            "can_answer": True,
            "next_step": "llm",
            "reliability": reliability,
            "context": context,
            "answer": None,
            "sources": context["sources"],
        }

    def _select_context_evidence(
        self,
        rerank_results: Sequence[Mapping[str, Any]] | None,
        top_score: float | None,
    ) -> list[Mapping[str, Any]]:
        """Keep only strong, unique evidence that can actually enter the prompt."""
        if not rerank_results or top_score is None:
            return []

        minimum_score = max(
            self.answer_reliability.threshold,
            top_score - self.context_score_gap,
        )
        selected: list[Mapping[str, Any]] = []
        seen_content: set[str] = set()
        seen_chunks: set[tuple[str, object]] = set()

        for candidate in rerank_results:
            if not isinstance(candidate, Mapping):
                continue
            content = candidate.get("content")
            if content is None:
                content = candidate.get("context")
            if not isinstance(content, str) or not content.strip():
                continue
            try:
                score = float(candidate.get("rerank_score"))
            except (TypeError, ValueError):
                continue
            if not isfinite(score) or score < minimum_score:
                continue

            normalized_content = " ".join(content.split()).casefold()
            index = candidate.get("chunk_index", candidate.get("index"))
            chunk_key = (str(candidate.get("source") or ""), index)
            has_chunk_identity = bool(chunk_key[0]) and chunk_key[1] is not None
            if normalized_content in seen_content or (
                has_chunk_identity and chunk_key in seen_chunks
            ):
                continue

            selected.append(candidate)
            seen_content.add(normalized_content)
            if has_chunk_identity:
                seen_chunks.add(chunk_key)
            if len(selected) >= self.context_max_chunks:
                break

        return selected

    def run(
        self,
        query: str,
        rerank_results: Sequence[Mapping[str, Any]] | None,
    ) -> PipelineResult:
        """Run the complete local chain and call DeepSeek only after the gate."""
        prepared = self.run_after_rerank(rerank_results)
        if not prepared["can_answer"]:
            return prepared

        context = prepared["context"]
        if context is None:
            return {
                **prepared,
                "can_answer": False,
                "next_step": "human_handoff",
                "answer": None,
                "sources": [],
            }

        answer = self.generator.generate(query, context["context"])
        return {
            **prepared,
            "next_step": "complete",
            "answer": answer,
            "sources": context["sources"],
        }
