"""Vector similarity retrieval."""

from typing import Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, ScoredPoint

from app.services.embedding_service import EmbeddingService
from config.settings import settings


class VectorSearch:
    """Search the configured Qdrant collection with local query embeddings."""

    def __init__(
        self,
        client: QdrantClient | None = None,
        embedding_service: EmbeddingService | None = None,
        collection_name: str | None = None,
        corpus_id: str | None = None,
    ) -> None:
        if settings is None:
            raise RuntimeError("Project settings are unavailable")

        self.collection_name = collection_name or settings.qdrant_collection
        self.corpus_id = str(
            corpus_id if corpus_id is not None else settings.knowledge_corpus_id
        ).strip()
        if not self.corpus_id:
            raise ValueError("corpus_id must not be empty")
        self.client = client if client is not None else QdrantClient(url=settings.qdrant_url)
        self.embedding_service = (
            embedding_service if embedding_service is not None else EmbeddingService()
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        query_filter: Filter | None = None,
    ) -> Sequence[ScoredPoint]:
        """Embed a user query and return the top matching Qdrant points."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_vector = self.embedding_service.embed_query(query)
        query_kwargs = {
            "collection_name": self.collection_name,
            "query": query_vector,
            "limit": top_k,
            "with_payload": True,
        }
        query_kwargs["query_filter"] = self._corpus_filter(query_filter)
        result = self.client.query_points(**query_kwargs)
        return result.points

    def verify_collection(self) -> None:
        """Verify that the configured collection exists and is reachable."""

        if not self.client.collection_exists(self.collection_name):
            raise RuntimeError(f"Qdrant collection is unavailable: {self.collection_name}")

    def _corpus_filter(self, query_filter: Filter | None) -> Filter:
        """Require this retriever's corpus without discarding caller scope filters."""

        corpus_condition = FieldCondition(
            key="corpus_id",
            match=MatchValue(value=self.corpus_id),
        )
        if query_filter is None:
            return Filter(must=[corpus_condition])
        return Filter(must=[corpus_condition, query_filter])


def xianyu_item_filter(item_id: str) -> Filter:
    """Allow common seller rules and only the selected item's documents."""

    normalized_id = str(item_id or "").strip()
    if not normalized_id:
        raise ValueError("item_id must not be empty")
    return Filter(
        should=[
            FieldCondition(key="scope", match=MatchValue(value="common")),
            Filter(
                must=[
                    FieldCondition(key="scope", match=MatchValue(value="item")),
                    FieldCondition(key="item_id", match=MatchValue(value=normalized_id)),
                ]
            ),
        ]
    )


def xianyu_common_filter() -> Filter:
    """Allow only seller-wide rules when no concrete item is selected."""

    return Filter(
        must=[FieldCondition(key="scope", match=MatchValue(value="common"))]
    )
