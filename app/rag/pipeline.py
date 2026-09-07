"""RAG pipeline from reranked evidence to a grounded answer."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypedDict

from app.generation.deepseek import DeepSeekGenerator
from app.rag.answerability import AnswerReliability, ReliabilityResult
from app.rag.context_builder import (
    ContextBuildResult,
    ContextBuilder,
    SourceReference,
)


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
    ) -> None:
        self.answer_reliability = answer_reliability or AnswerReliability()
        self.context_builder = context_builder or ContextBuilder()
        self.generator = generator or DeepSeekGenerator()

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

        context = self.context_builder.build(rerank_results)
        return {
            "can_answer": True,
            "next_step": "llm",
            "reliability": reliability,
            "context": context,
            "answer": None,
            "sources": context["sources"],
        }

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
