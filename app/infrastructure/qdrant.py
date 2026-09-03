"""Qdrant client and collection infrastructure."""

from typing import Sequence
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.ingestion.chunker import Chunk
from config.settings import settings


class QdrantStore:
    """Write chunk vectors and metadata to the configured Qdrant collection."""

    def __init__(self, client: QdrantClient | None = None) -> None:
        if settings is None:
            raise RuntimeError("Project settings are unavailable")

        self.collection_name = settings.qdrant_collection
        self.client = client or QdrantClient(url=settings.qdrant_url)

    def build_points(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> list[PointStruct]:
        """Combine chunks and vectors into Qdrant points."""
        if len(chunks) != len(vectors):
            raise ValueError("The number of chunks must match the number of vectors")

        if not chunks:
            return []

        vector_size = len(vectors[0])
        if vector_size == 0:
            raise ValueError("Vectors must not be empty")

        if any(len(vector) != vector_size for vector in vectors):
            raise ValueError("All vectors must have the same dimension")

        return [
            PointStruct(
                id=str(uuid5(NAMESPACE_URL, f"{chunk.source}:{chunk_index}")),
                vector=list(vector),
                payload={
                    "content": chunk.content,
                    "source": chunk.source,
                    "chunk_index": chunk_index,
                },
            )
            for chunk_index, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]

    def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        """Create the collection when needed and write all chunk points."""
        points = self.build_points(chunks, vectors)
        if not points:
            return

        vector_size = len(points[0].vector)
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
