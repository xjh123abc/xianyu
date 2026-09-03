"""Vector similarity retrieval."""

from typing import Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint

from app.services.embedding_service import EmbeddingService
from config.settings import settings


class VectorSearch:
    """Search the configured Qdrant collection with local query embeddings."""

    def __init__(
        self,
        client: QdrantClient | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        if settings is None:
            raise RuntimeError("Project settings are unavailable")

        self.collection_name = settings.qdrant_collection
        self.client = client if client is not None else QdrantClient(url=settings.qdrant_url)
        self.embedding_service = (
            embedding_service if embedding_service is not None else EmbeddingService()
        )

    def search(self, query: str, top_k: int = 5) -> Sequence[ScoredPoint]:
        """Embed a user query and return the top matching Qdrant points."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_vector = self.embedding_service.embed_query(query)
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        return result.points
