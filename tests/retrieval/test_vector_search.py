from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.ingestion.chunker import Chunk, chunk_document
from app.ingestion.loader import load_txt
from app.infrastructure.qdrant import QdrantStore
from app.retrieval.vector_search import VectorSearch
from app.services.embedding_service import EmbeddingService
from config.settings import settings


class StaticEmbeddingService:
    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0]


def test_query_returns_top_five_points_from_qdrant() -> None:
    document = load_txt("data/raw/ingestion_test.txt")
    chunks = chunk_document(document, chunk_size=100)
    embedding_service = EmbeddingService()
    vectors = embedding_service.embed_chunks(chunks)
    client = QdrantClient(location=":memory:")

    try:
        QdrantStore(client=client).upsert_chunks(chunks, vectors)
        search = VectorSearch(client=client, embedding_service=embedding_service)

        points = search.search("How long does shipping take?")

        assert settings is not None
        assert search.collection_name == settings.qdrant_collection
        assert len(points) == 5
        assert all(point.payload is not None for point in points)
        assert all(
            {"content", "source", "chunk_index"}.issubset(point.payload)
            for point in points
        )
        assert all(point.score is not None for point in points)
        assert len({str(point.id) for point in points}) == 5
    finally:
        client.close()


def test_query_filters_out_points_from_another_corpus() -> None:
    client = QdrantClient(location=":memory:")
    current = QdrantStore(client=client, collection_name="shared", corpus_id="current")
    foreign = QdrantStore(client=client, collection_name="shared", corpus_id="foreign")

    try:
        current.upsert_chunks([Chunk("current policy", "current.md")], [[1.0, 0.0]])
        foreign.upsert_chunks([Chunk("foreign policy", "foreign.md")], [[1.0, 0.0]])
        search = VectorSearch(
            client=client,
            embedding_service=StaticEmbeddingService(),
            collection_name="shared",
            corpus_id="current",
        )

        points = search.search("policy")

        assert [point.payload["content"] for point in points] == ["current policy"]
        assert all(point.payload["corpus_id"] == "current" for point in points)
    finally:
        client.close()


def test_query_combines_corpus_and_caller_scope_filters() -> None:
    client = QdrantClient(location=":memory:")
    current = QdrantStore(client=client, collection_name="shared", corpus_id="current")
    foreign = QdrantStore(client=client, collection_name="shared", corpus_id="foreign")

    try:
        current.upsert_chunks(
            [
                Chunk("current target", "target.md", scope="item", item_id="TARGET"),
                Chunk("current other", "other.md", scope="item", item_id="OTHER"),
            ],
            [[1.0, 0.0], [1.0, 0.0]],
        )
        foreign.upsert_chunks(
            [Chunk("foreign target", "foreign.md", scope="item", item_id="TARGET")],
            [[1.0, 0.0]],
        )
        search = VectorSearch(
            client=client,
            embedding_service=StaticEmbeddingService(),
            collection_name="shared",
            corpus_id="current",
        )
        target_filter = Filter(
            must=[FieldCondition(key="item_id", match=MatchValue(value="TARGET"))]
        )

        points = search.search("policy", query_filter=target_filter)

        assert [point.payload["content"] for point in points] == ["current target"]
        assert all(point.payload["corpus_id"] == "current" for point in points)
        assert all(point.payload["item_id"] == "TARGET" for point in points)
    finally:
        client.close()
