from qdrant_client import QdrantClient

from app.ingestion.chunker import chunk_document
from app.ingestion.loader import load_txt
from app.infrastructure.qdrant import QdrantStore
from app.retrieval.vector_search import VectorSearch
from app.services.embedding_service import EmbeddingService
from config.settings import settings


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
