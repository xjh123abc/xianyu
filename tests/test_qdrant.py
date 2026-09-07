from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient

from app.ingestion.chunker import Chunk, chunk_document
from app.ingestion.loader import (
    configured_knowledge_base_path,
    load_configured_knowledge_base,
    load_txt,
)
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


def test_formal_ingestion_replaces_legacy_absolute_source_points() -> None:
    chunks = load_configured_knowledge_base()
    knowledge_base_path = configured_knowledge_base_path()
    legacy_chunk = Chunk(
        content=chunks[0].content,
        source=str(knowledge_base_path / chunks[0].source),
    )
    client = QdrantClient(location=":memory:")
    store = QdrantStore(client=client)

    try:
        store.upsert_chunks([legacy_chunk], [[0.1, 0.2]])
        store.upsert_chunks(chunks, [[0.1, 0.2] for _ in chunks])

        points, _ = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=len(chunks) + 1,
            with_payload=True,
        )
        assert len(points) == len(chunks)
        assert all(not Path(point.payload["source"]).is_absolute() for point in points)
        assert all(
            point.payload["source"] == chunks[point.payload["chunk_index"]].source
            for point in points
        )
        expected_id = str(uuid5(NAMESPACE_URL, f"{chunks[0].source}:0"))
        assert any(point.id == expected_id for point in points)
    finally:
        client.close()
