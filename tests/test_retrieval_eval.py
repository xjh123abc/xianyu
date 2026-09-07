import json
import shutil
from pathlib import Path

import pytest

from app.ingestion.chunker import Chunk
from app.retrieval.bm25 import BM25Search
from app.retrieval.reranker import Reranker
from config.settings import settings
from eval.retrieval_eval import run_evaluation


class FakeDenseSearch:
    def __init__(self, results):
        self.results = results
        self.queries = []

    def search(self, query, *, top_k):
        self.queries.append((query, top_k))
        return self.results[:top_k]


class FakeReranker:
    def __init__(self):
        self.candidates = []

    def rerank(self, query, candidates, *, top_k):
        self.candidates.append((query, list(candidates), top_k))
        return [
            {**dict(candidate), "rerank_score": 0.9 - index * 0.1}
            for index, candidate in enumerate(candidates[:top_k])
        ]


@pytest.fixture
def local_eval_dir():
    path = Path(__file__).parent / ".retrieval-eval-test-artifacts"
    if path.exists():
        shutil.rmtree(path)
    path.mkdir()
    try:
        yield path
    finally:
        if path.exists():
            shutil.rmtree(path)


def test_evaluation_fuses_recorded_dense_and_bm25_results(
    local_eval_dir: Path,
    monkeypatch,
) -> None:
    input_path = local_eval_dir / "cases.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "id": "case-1",
                "question": "shipping",
                "expected_sources": ["shipping.md"],
                "expected_keywords": ["shipping"],
                "answerable": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = local_eval_dir / "results" / "retrieval_results.jsonl"
    monkeypatch.setattr(
        "eval.retrieval_eval.load_configured_knowledge_base",
        lambda: [
            Chunk("shipping instructions", "shipping.md"),
            Chunk("refund instructions", "refund.md"),
        ],
    )
    monkeypatch.setattr("eval.retrieval_eval.ensure_knowledge_base_in_sync", lambda chunks: {})
    monkeypatch.setattr(settings, "dense_top_k", 1)
    monkeypatch.setattr(settings, "bm25_top_k", 1)
    monkeypatch.setattr(settings, "reranker_top_k", 1)

    dense = FakeDenseSearch(
        [
            {
                "content": "dense content",
                "source": "dense.md",
                "chunk_index": 0,
                "score": 0.8,
            }
        ]
    )
    reranker = FakeReranker()

    assert run_evaluation(
        input_path,
        output_path,
        vector_search=dense,
        bm25_search=BM25Search(
            [
                Chunk("shipping instructions", "shipping.md"),
                Chunk("refund instructions", "refund.md"),
            ]
        ),
        reranker=reranker,
    ) == 1

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["answerable"] is False
    assert result["dense"][0]["source"] == "dense.md"
    assert result["bm25"][0]["source"] == "shipping.md"
    assert result["rrf"][0]["source"] in {"dense.md", "shipping.md"}
    assert reranker.candidates[0][1] == [
        {
            "content": "dense content",
            "source": "dense.md",
            "chunk_index": 0,
            "score": 1 / 61,
        },
        {
            "content": "shipping instructions",
            "source": "shipping.md",
            "chunk_index": 0,
            "score": 1 / 61,
        },
    ][: len(reranker.candidates[0][1])]
