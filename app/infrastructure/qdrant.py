"""Qdrant client and collection infrastructure."""

import hashlib
from pathlib import Path
from typing import Sequence
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.ingestion.chunker import Chunk
from config.settings import settings


class QdrantStore:
    """Write chunk vectors and metadata to the configured Qdrant collection."""

    def __init__(
        self,
        client: QdrantClient | None = None,
        *,
        collection_name: str | None = None,
        knowledge_base_path: str | Path | None = None,
        corpus_id: str | None = None,
    ) -> None:
        if settings is None:
            raise RuntimeError("Project settings are unavailable")

        self.collection_name = collection_name or settings.qdrant_collection
        self.knowledge_base_path = (
            Path(knowledge_base_path).resolve() if knowledge_base_path is not None else None
        )
        if corpus_id is not None:
            resolved_corpus_id = corpus_id.strip()
        elif self.knowledge_base_path is None:
            resolved_corpus_id = str(settings.knowledge_corpus_id).strip()
        else:
            path_digest = hashlib.sha256(
                self.knowledge_base_path.as_posix().casefold().encode("utf-8")
            ).hexdigest()[:12]
            resolved_corpus_id = f"custom-{path_digest}"
        if not resolved_corpus_id:
            raise ValueError("corpus_id must not be empty")
        self.corpus_id = resolved_corpus_id
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
                "corpus_id": self.corpus_id,
            }
            if chunk.scope is not None:
                payload["scope"] = chunk.scope
            if chunk.item_id is not None:
                payload["item_id"] = chunk.item_id
            points.append(
                PointStruct(
                    id=str(
                        uuid5(
                            NAMESPACE_URL,
                            f"{self.corpus_id}:{chunk.source}:{chunk_index}",
                        )
                    ),
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
        """Create the collection and replace points from this corpus only."""
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
        """Delete stale points carrying this store's explicit corpus identity."""
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
                payload = record.payload or {}
                if payload.get("corpus_id") == self.corpus_id:
                    legacy_point_ids.append(str(record.id))

            if offset is None:
                break

        if legacy_point_ids:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=legacy_point_ids,
                wait=True,
            )
