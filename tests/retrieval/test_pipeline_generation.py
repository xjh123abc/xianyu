from app.rag.answerability import AnswerReliability
from app.rag.pipeline import RAGPipeline


class FakeGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def generate(self, query: str, context: str) -> str:
        self.calls.append((query, context))
        return "基于资料生成的答案"


def test_pipeline_passes_query_and_context_to_generator_and_returns_sources() -> None:
    generator = FakeGenerator()
    pipeline = RAGPipeline(
        answer_reliability=AnswerReliability(threshold=0.5),
        generator=generator,
    )

    result = pipeline.run(
        "商品破损怎么办？",
        [
            {
                "content": "请保留包装并拍摄照片。",
                "source": "09_after_sales.md",
                "chunk_index": 0,
                "rerank_score": 0.8,
            }
        ],
    )

    assert result["can_answer"] is True
    assert result["next_step"] == "complete"
    assert result["answer"] == "基于资料生成的答案"
    assert result["sources"] == [{"source": "09_after_sales.md", "index": 0}]
    assert generator.calls == [("商品破损怎么办？", "请保留包装并拍摄照片。")]


def test_pipeline_does_not_call_generator_when_reliability_fails() -> None:
    generator = FakeGenerator()
    pipeline = RAGPipeline(
        answer_reliability=AnswerReliability(threshold=0.5),
        generator=generator,
    )

    result = pipeline.run(
        "商品破损怎么办？",
        [{"content": "无关资料", "rerank_score": 0.2}],
    )

    assert result["can_answer"] is False
    assert result["next_step"] == "clarify"
    assert result["answer"] is None
    assert result["sources"] == []
    assert generator.calls == []
