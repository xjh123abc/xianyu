"""Hybrid keyword and vector retrieval."""

from collections.abc import Callable, Mapping, Sequence
from typing import TypedDict

from qdrant_client.models import Filter

from app.retrieval.bm25 import BM25Search
from app.retrieval.vector_search import VectorSearch


class HybridResult(TypedDict):
    """A chunk returned by the fused retriever."""

    content: str
    score: float
    source: str
    chunk_index: int
    scope: str | None
    item_id: str | None


class HybridSearch:
    """Fuse dense and BM25 rankings with Reciprocal Rank Fusion."""

    def __init__(
        self,
        vector_search: VectorSearch | None = None,
        bm25_search: BM25Search | None = None,
        *,
        dense_search: VectorSearch | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.vector_search = dense_search if dense_search is not None else vector_search
        self.bm25_search = bm25_search
        if self.vector_search is None or self.bm25_search is None:
            raise ValueError("vector_search and bm25_search are required")
        if rrf_k < 0:
            raise ValueError("rrf_k must not be negative")
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int | None = 5,
        *,
        dense_top_k: int | None = None,
        bm25_top_k: int | None = None,
        query_filter: Filter | None = None,
        bm25_filter: Callable[[object], bool] | None = None,
    ) -> list[HybridResult]:
        """Return one ranking built from the two independent result rankings."""
        dense_limit = dense_top_k if dense_top_k is not None else top_k
        bm25_limit = bm25_top_k if bm25_top_k is not None else top_k
        if dense_limit is None or dense_limit <= 0:
            raise ValueError("dense_top_k must be greater than zero")
        if bm25_limit is None or bm25_limit <= 0:
            raise ValueError("bm25_top_k must be greater than zero")
        output_limit = top_k if top_k is not None else max(dense_limit, bm25_limit)
        if output_limit <= 0:
            raise ValueError("top_k must be greater than zero")

        if query_filter is None:
            dense_results = self.vector_search.search(query, top_k=dense_limit)
        else:
            dense_results = self.vector_search.search(
                query, top_k=dense_limit, query_filter=query_filter
            )
        if bm25_filter is None:
            bm25_results = self.bm25_search.search(query, top_k=bm25_limit)
        else:
            bm25_results = self.bm25_search.search(
                query, top_k=bm25_limit, filter_fn=bm25_filter
            )
        return self.fuse(dense_results, bm25_results, top_k=output_limit)

    def fuse(
        self,
        dense_results: Sequence[object],
        bm25_results: Sequence[object],
        *,
        top_k: int = 5,
    ) -> list[HybridResult]:
        """Fuse already retrieved Dense and BM25 results with RRF.

        Keeping this operation separate lets callers record the exact two
        retrieval results that were used as the RRF inputs.
        """
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        merged: dict[tuple[str, int], HybridResult] = {}

        for rank, result in enumerate(dense_results, start=1):
            self._add_rank(merged, self._read_result(result), rank)
        for rank, result in enumerate(bm25_results, start=1):
            self._add_rank(merged, self._read_result(result), rank)

        return sorted(
            merged.values(),
            key=lambda result: (-result["score"], result["source"], result["chunk_index"]),
        )[:top_k]

    def _add_rank(
        self,
        merged: dict[tuple[str, int], HybridResult],
        result: HybridResult,
        rank: int,
    ) -> None:
        """Add one rank contribution without using its retrieval score."""
        key = (result["source"], result["chunk_index"])
        contribution = 1.0 / (self.rrf_k + rank)
        if key not in merged:
            merged[key] = {
                "content": result["content"],
                "score": contribution,
                "source": result["source"],
                "chunk_index": result["chunk_index"],
            }
            if result.get("scope") is not None:
                merged[key]["scope"] = result["scope"]
            if result.get("item_id") is not None:
                merged[key]["item_id"] = result["item_id"]
        else:
            merged[key]["score"] += contribution

    @staticmethod
    def _read_result(result: object) -> HybridResult:
        """Normalize a BM25 dictionary or a Qdrant scored point."""
        if isinstance(result, Mapping):
            payload = result
        else:
            payload = getattr(result, "payload", None) or {}

        normalized: HybridResult = {
            "content": str(payload["content"]),
            "score": float(payload.get("score", getattr(result, "score", 0.0))),
            "source": str(payload["source"]),
            "chunk_index": int(payload["chunk_index"]),
        }
        if payload.get("scope") is not None:
            normalized["scope"] = str(payload["scope"])
        if payload.get("item_id") is not None:
            normalized["item_id"] = str(payload["item_id"])
        return normalized
