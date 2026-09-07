from unittest.mock import Mock

import pytest

from app.rag.answerability import AnswerReliability
from app.rag.pipeline import RAGPipeline


def test_reliability_reads_threshold_from_settings(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.rag.answerability.settings.answer_reliability_threshold",
        0.8,
    )

    result = AnswerReliability().evaluate(
        [{"content": "usable evidence", "rerank_score": 0.8}]
    )

    assert result == {
        "can_answer": True,
        "next_step": "context_builder",
        "reason": "sufficient_evidence",
        "top_rerank_score": 0.8,
        "threshold": 0.8,
    }


def test_reliability_uses_first_valid_rerank_result_without_sorting() -> None:
    result = AnswerReliability(threshold=0.7).evaluate(
        [
            {"content": "top-ranked evidence", "rerank_score": 0.6},
            {"content": "lower-ranked evidence", "rerank_score": 0.95},
        ]
    )

    assert result["can_answer"] is False
    assert result["top_rerank_score"] == 0.6
    assert result["reason"] == "score_below_threshold"


@pytest.mark.parametrize(
    "rerank_results, reason",
    [
        ([], "no_results"),
        ([{"content": "   ", "rerank_score": 0.9}], "no_valid_content"),
        ([{"content": "evidence"}], "missing_rerank_score"),
        ([{"content": "evidence", "rerank_score": "not-a-score"}], "missing_rerank_score"),
    ],
)
def test_reliability_treats_invalid_evidence_as_handoff(
    rerank_results,
    reason,
) -> None:
    result = AnswerReliability(threshold=0.5).evaluate(rerank_results)

    assert result["can_answer"] is False
    assert result["next_step"] == "human_handoff"
    assert result["reason"] == reason
    assert result["top_rerank_score"] is None


def test_pipeline_builds_context_only_after_reliability_passes() -> None:
    context_builder = Mock()
    context_builder.build.return_value = {
        "context": "usable evidence",
        "sources": [{"source": "policy.md", "index": 0}],
    }
    pipeline = RAGPipeline(
        answer_reliability=AnswerReliability(threshold=0.5),
        context_builder=context_builder,
    )

    blocked = pipeline.run_after_rerank(
        [{"content": "weak evidence", "rerank_score": 0.4}]
    )
    allowed = pipeline.run_after_rerank(
        [{
            "content": "strong evidence",
            "source": "policy.md",
            "chunk_index": 0,
            "rerank_score": 0.8,
        }]
    )

    assert blocked["can_answer"] is False
    assert blocked["next_step"] == "human_handoff"
    assert blocked["context"] is None
    context_builder.build.assert_called_once()

    assert allowed["can_answer"] is True
    assert allowed["next_step"] == "llm"
    assert allowed["context"] == {
        "context": "usable evidence",
        "sources": [{"source": "policy.md", "index": 0}],
    }


def test_pipeline_does_not_build_context_for_empty_results() -> None:
    context_builder = Mock()
    pipeline = RAGPipeline(
        answer_reliability=AnswerReliability(threshold=0.5),
        context_builder=context_builder,
    )

    result = pipeline.run_after_rerank([])

    assert result["can_answer"] is False
    assert result["next_step"] == "human_handoff"
    assert result["context"] is None
    context_builder.build.assert_not_called()
