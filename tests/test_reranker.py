from pathlib import Path

import pytest

from app.retrieval import reranker as reranker_module
from app.retrieval.reranker import Reranker


class FakeScorer:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.pairs = None

    def predict(self, pairs):
        self.pairs = pairs
        return self.scores


def test_rerank_scores_orders_top_k_and_preserves_metadata() -> None:
    scorer = FakeScorer([0.2, 0.95, 0.6])
    candidates = [
        {
            "content": "first chunk",
            "source": "a.md",
            "chunk_index": 0,
            "vector_score": 0.7,
            "bm25_score": 1.2,
            "bm25_rank": 2,
            "rrf_score": 0.03,
            "metadata": {"section": "one"},
        },
        {
            "content": "second chunk",
            "source": "b.md",
            "chunk_index": 1,
            "vector_score": 0.8,
            "bm25_score": 0.9,
            "bm25_rank": 1,
            "rrf_score": 0.04,
            "metadata": {"section": "two"},
        },
        {
            "content": "third chunk",
            "source": "c.md",
            "chunk_index": 2,
            "rrf_score": 0.02,
        },
    ]

    results = Reranker(scorer=scorer).rerank("test query", candidates, top_k=2)

    assert scorer.pairs == [
        ("test query", "first chunk"),
        ("test query", "second chunk"),
        ("test query", "third chunk"),
    ]
    assert [result["content"] for result in results] == [
        "second chunk",
        "third chunk",
    ]
    assert [result["rerank_score"] for result in results] == [0.95, 0.6]
    assert results[0] == {
        **candidates[1],
        "rerank_score": 0.95,
    }
    assert "rerank_score" not in candidates[0]


def test_rerank_skips_candidates_without_content() -> None:
    scorer = FakeScorer([0.4])
    candidates = [
        {"source": "missing.md", "chunk_index": 0, "rrf_score": 0.1},
        {"content": "", "source": "empty.md", "chunk_index": 1},
        {"content": "valid", "source": "valid.md", "chunk_index": 2},
    ]

    results = Reranker(scorer=scorer).rerank("query", candidates)

    assert results == [
        {
            "content": "valid",
            "source": "valid.md",
            "chunk_index": 2,
            "rerank_score": 0.4,
        }
    ]
    assert scorer.pairs == [("query", "valid")]


def test_rerank_handles_empty_candidates_without_predicting() -> None:
    scorer = FakeScorer([])

    assert Reranker(scorer=scorer).rerank("query", []) == []
    assert scorer.pairs is None


def test_rerank_rejects_non_positive_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        Reranker(scorer=FakeScorer([])).rerank("query", [], top_k=0)


def test_model_load_uses_configured_local_path_without_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    local_model_path = Path(__file__).parent

    def fake_cross_encoder(model_path, **kwargs):
        calls.append((model_path, kwargs))
        return object()

    monkeypatch.setattr(
        reranker_module.settings,
        "reranker_model_path",
        str(local_model_path),
    )
    monkeypatch.setattr(reranker_module, "CrossEncoder", fake_cross_encoder)

    Reranker()

    assert calls == [(str(local_model_path), {"local_files_only": True})]


def test_model_load_rejects_missing_path(monkeypatch: pytest.MonkeyPatch) -> None:
    missing_path = Path(__file__).parent / "missing-reranker"
    monkeypatch.setattr(
        reranker_module.settings,
        "reranker_model_path",
        str(missing_path),
    )

    with pytest.raises(FileNotFoundError, match="Reranker model path"):
        Reranker()


def test_configured_local_reranker_scores_and_reranks_candidates() -> None:
    query = "商品不满意怎么退货？"
    candidates = [
        {
            "content": "如果商品已经收到，并且符合退货条件，可以在订单详情页申请退货退款。提交申请后按系统提供的地址寄回商品。",
            "source": "09_after_sales.md",
            "chunk_index": 0,
            "rrf_score": 0.03,
            "metadata": {"topic": "退货"},
        },
        {
            "content": "商品发出后，普通跨省订单通常2-5天送达，偏远地区通常4-7天送达。",
            "source": "02_shipping.md",
            "chunk_index": 1,
            "rrf_score": 0.02,
            "metadata": {"topic": "物流"},
        },
        {
            "content": "如果订单发生退款，对应订单获得的积分可能会被扣回。积分通常在确认收货后发放。",
            "source": "06_membership.md",
            "chunk_index": 2,
            "rrf_score": 0.01,
            "metadata": {"topic": "会员积分"},
        },
    ]

    results = Reranker(scorer=FakeScorer([0.9, 0.2, 0.1])).rerank(query, candidates, top_k=2)

    assert len(results) == 2
    assert all("rerank_score" in result for result in results)
    assert results[0]["source"] == "09_after_sales.md"
    assert results[0]["rerank_score"] >= results[1]["rerank_score"]
    assert results[0]["chunk_index"] == 0
    assert results[0]["rrf_score"] == 0.03
    assert results[0]["metadata"] == {"topic": "退货"}
