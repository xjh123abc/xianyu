"""Qdrant client and collection infrastructure."""

from pathlib import Path
from typing import Sequence
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.ingestion.chunker import Chunk
from app.ingestion.loader import canonical_source, configured_knowledge_base_path
from config.settings import settings


class QdrantStore:
    """Write chunk vectors and metadata to the configured Qdrant collection."""

    def __init__(
        self,
        client: QdrantClient | None = None,
        *,
        collection_name: str | None = None,
        knowledge_base_path: str | Path | None = None,
    ) -> None:
        if settings is None:
            raise RuntimeError("Project settings are unavailable")

        self.collection_name = collection_name or settings.qdrant_collection
        self.knowledge_base_path = (
            Path(knowledge_base_path).resolve() if knowledge_base_path is not None else None
        )
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

        points: list[PointStruct] = []
        for chunk_index, (chunk, vector) in enumerate(zip(chunks, vectors)):
            payload = {
                "content": chunk.content,
                "source": chunk.source,
                "chunk_index": chunk_index,
            }
            if chunk.scope is not None:
                payload["scope"] = chunk.scope
            if chunk.item_id is not None:
                payload["item_id"] = chunk.item_id
            points.append(
                PointStruct(
                    id=str(uuid5(NAMESPACE_URL, f"{chunk.source}:{chunk_index}")),
                    vector=list(vector),
                    payload=payload,
                )
            )
        return points

    def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        """Create the collection and replace legacy formal-KB points."""
        points = self.build_points(chunks, vectors)
        if not points:
            if self.client.collection_exists(self.collection_name):
                self._remove_existing_knowledge_base_points()
            return

        vector_size = len(points[0].vector)
        collection_exists = self.client.collection_exists(self.collection_name)
        if not collection_exists:
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
        if collection_exists:
            self._remove_existing_knowledge_base_points(
                excluded_ids={str(point.id) for point in points}
            )

    def _remove_existing_knowledge_base_points(
        self,
        *,
        excluded_ids: set[str] | None = None,
    ) -> None:
        """Delete existing points belonging to the configured knowledge base."""
        knowledge_base_path = self.knowledge_base_path or configured_knowledge_base_path()
        excluded_ids = excluded_ids or set()
        legacy_point_ids: list[str] = []
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                if str(record.id) in excluded_ids:
                    continue
                source = (record.payload or {}).get("source")
                if not isinstance(source, str):
                    continue
                if not Path(source).is_absolute() and Path(source).suffix.lower() in {".md", ".txt"}:
                    legacy_point_ids.append(str(record.id))
                    continue
                try:
                    logical_source = canonical_source(source, knowledge_base_path)
                except ValueError:
                    continue
                if Path(logical_source).suffix.lower() in {".md", ".txt"}:
                    legacy_point_ids.append(str(record.id))

            if offset is None:
                break

        if legacy_point_ids:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=legacy_point_ids,
                wait=True,
            )
