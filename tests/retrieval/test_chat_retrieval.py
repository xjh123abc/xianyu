from qdrant_client import QdrantClient
from pathlib import Path

from app.api import chat as chat_api
from app.ingestion.chunker import chunk_document
from app.ingestion.loader import (
    configured_knowledge_base_path,
    load_configured_knowledge_base,
    load_txt,
)
from app.infrastructure.qdrant import QdrantStore
from app.main import app
from app.rag.answerability import AnswerReliability
from app.rag.pipeline import RAGPipeline
from app.retrieval.bm25 import BM25Search
from app.retrieval.hybrid_search import HybridSearch
from app.retrieval.reranker import Reranker
from app.retrieval.vector_search import VectorSearch
from app.services.chat_service import ChatService
from app.services.embedding_service import EmbeddingService
from config.settings import settings
from fastapi.testclient import TestClient


client = TestClient(app)


class FakeScorer:
    def predict(self, pairs):
        return [1.0 for _ in pairs]


class FakeGenerator:
    def generate(self, query: str, context: str) -> str:
        return "基于检索资料生成的测试答案"


class FakeDenseSearch:
    def __init__(self, points):
        self.points = points

    def search(self, query: str, top_k: int = 5):
        return self.points[:top_k]


class FakePoint:
    def __init__(self, chunk, chunk_index: int):
        self.payload = {
            "content": chunk.content,
            "source": chunk.source,
            "chunk_index": chunk_index,
        }
        self.score = 1.0


def test_chat_retrieval_chain_returns_top_five_results(monkeypatch) -> None:
    document = load_txt("tests/fixtures/ingestion_test.txt")
    chunks = chunk_document(document, chunk_size=100)
    embedding_service = EmbeddingService()
    vectors = embedding_service.embed_chunks(chunks)
    qdrant_client = QdrantClient(location=":memory:")

    try:
        QdrantStore(client=qdrant_client).upsert_chunks(chunks, vectors)
        vector_search = VectorSearch(
            client=qdrant_client,
            embedding_service=embedding_service,
        )
        monkeypatch.setattr(
            chat_api,
            "chat_service",
            ChatService(
                vector_search,
                bm25_search=BM25Search(chunks),
                reranker=Reranker(scorer=FakeScorer()),
                    rag_pipeline=RAGPipeline(
                        answer_reliability=AnswerReliability(threshold=0.5),
                        generator=FakeGenerator(),
                    ),
            ),
        )

        response = client.post(
            "/chat",
            json={"query": "退款一般多久才能到账？", "chat_id": "chat_rag_001"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["query"] == "退款一般多久才能到账？"
        assert body["chat_id"] == "chat_rag_001"
        assert body["answer"] == "基于检索资料生成的测试答案"
        assert body["can_answer"] is True
        assert body["next_step"] == "complete"
        assert len(body["results"]) == 5
        assert all(
            set(result) == {"content", "score", "source", "chunk_index"}
            for result in body["results"]
        )
        assert all(isinstance(result["content"], str) for result in body["results"])
        assert all(isinstance(result["score"], float) for result in body["results"])
        assert all(isinstance(result["source"], str) for result in body["results"])
        assert all(isinstance(result["chunk_index"], int) for result in body["results"])
        assert settings is not None
        assert all(
            result["source"].endswith("ingestion_test.txt")
            for result in body["results"]
        )
    finally:
        qdrant_client.close()


def test_formal_bm25_and_hybrid_exclude_data_raw_sources() -> None:
    chunks = load_configured_knowledge_base()
    formal_path = configured_knowledge_base_path()
    formal_sources = {
        path.relative_to(formal_path).as_posix()
        for path in formal_path.iterdir()
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}
    }
    bm25_search = BM25Search(chunks)
    bm25_results = bm25_search.search("refund", top_k=5)
    top_bm25 = bm25_results[0]
    dense_points = [
        FakePoint(chunks[top_bm25["chunk_index"]], top_bm25["chunk_index"])
    ]

    results = HybridSearch(
        FakeDenseSearch(dense_points),
        bm25_search,
    ).search("refund", top_k=5)

    assert results
    assert all(
        result["source"] in formal_sources
        for result in results
    )
    assert all(
        "data/raw" not in result["source"].replace("\\", "/")
        for result in results
    )
    assert all(not Path(result["source"]).is_absolute() for result in results)
    fused_top_result = next(
        result
        for result in results
        if (result["source"], result["chunk_index"])
        == (top_bm25["source"], top_bm25["chunk_index"])
    )
    assert fused_top_result["score"] == 2 / (60 + 1)

    qdrant_client = QdrantClient(location=":memory:")
    try:
        qdrant_keys = [
            (point.payload["source"], point.payload["chunk_index"])
            for point in QdrantStore(client=qdrant_client).build_points(
                chunks,
                [[0.1, 0.2] for _ in chunks],
            )
        ]
        assert all(
            (result["source"], result["chunk_index"]) in qdrant_keys
            for result in results
        )
    finally:
        qdrant_client.close()
