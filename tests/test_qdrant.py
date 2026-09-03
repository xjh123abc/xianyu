from qdrant_client import QdrantClient

from app.ingestion.chunker import chunk_document
from app.ingestion.loader import load_txt
from app.infrastructure.qdrant import QdrantStore
from app.services.embedding_service import EmbeddingService
from config.settings import settings


def test_chunks_and_vectors_are_written_to_qdrant() -> None:
    document = load_txt("data/raw/ingestion_test.txt")
    chunks = chunk_document(document, chunk_size=100)
    vectors = EmbeddingService().embed_chunks(chunks)
    client = QdrantClient(location=":memory:")
    store = QdrantStore(client=client)

    try:
        store.upsert_chunks(chunks, vectors)

        assert settings is not None
        assert client.collection_exists(settings.qdrant_collection)
        collection = client.get_collection(settings.qdrant_collection)
        assert collection.config.params.vectors.size == len(vectors[0])

        points, _ = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=len(chunks),
            with_payload=True,
            with_vectors=True,
        )
        assert len(points) == len(chunks)

        points_by_index = sorted(points, key=lambda point: point.payload["chunk_index"])
        for chunk_index, (chunk, vector, point) in enumerate(
            zip(chunks, vectors, points_by_index)
        ):
            assert point.payload == {
                "content": chunk.content,
                "source": chunk.source,
                "chunk_index": chunk_index,
            }
            assert len(point.vector) == len(vector)

        store.upsert_chunks(chunks, vectors)
        points_after_reuse, _ = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=len(chunks) + 1,
        )
        assert len(points_after_reuse) == len(chunks)
    finally:
        client.close()
