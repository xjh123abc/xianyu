"""Minimal chat service implementation."""

from app.ingestion.loader import load_configured_knowledge_base
from app.rag.pipeline import RAGPipeline
from app.retrieval.bm25 import BM25Search
from app.retrieval.hybrid_search import HybridSearch
from app.retrieval.reranker import Reranker
from app.retrieval.vector_search import VectorSearch
from config.settings import settings


class ChatService:
    """Pass user queries to retrieval and format retrieved chunks."""

    def __init__(
        self,
        vector_search: VectorSearch | BM25Search | HybridSearch | None = None,
        bm25_search: BM25Search | None = None,
        reranker: Reranker | None = None,
        rag_pipeline: RAGPipeline | None = None,
    ) -> None:
        if isinstance(vector_search, (BM25Search, HybridSearch)) and bm25_search is None:
            self.vector_search = None
            self.bm25_search = vector_search if isinstance(vector_search, BM25Search) else None
            self.hybrid_search = vector_search if isinstance(vector_search, HybridSearch) else None
        else:
            self.vector_search = vector_search
            self.bm25_search = bm25_search
            self.hybrid_search = None
        self.reranker = reranker
        self.rag_pipeline = rag_pipeline

    def chat(self, query: str) -> dict[str, object]:
        """Return the original query and top five retrieved chunks."""
        if self.hybrid_search is not None:
            results = self.hybrid_search.search(
                query,
                top_k=max(settings.dense_top_k, settings.bm25_top_k),
                dense_top_k=settings.dense_top_k,
                bm25_top_k=settings.bm25_top_k,
            )
        elif self.vector_search is not None and self.bm25_search is not None:
            self.hybrid_search = HybridSearch(
                self.vector_search,
                self.bm25_search,
                rrf_k=settings.rrf_k,
            )
            results = self.hybrid_search.search(
                query,
                top_k=max(settings.dense_top_k, settings.bm25_top_k),
                dense_top_k=settings.dense_top_k,
                bm25_top_k=settings.bm25_top_k,
            )
        elif self.vector_search is not None:
            points = self.vector_search.search(query, top_k=settings.dense_top_k)
            results = []
            for point in points:
                payload = point.payload or {}
                results.append(
                    {
                        "content": payload["content"],
                        "score": point.score,
                        "source": payload["source"],
                        "chunk_index": payload["chunk_index"],
                    }
                )
        else:
            if self.bm25_search is None:
                chunks = load_configured_knowledge_base()
                from app.ingestion.pipeline import ensure_knowledge_base_in_sync

                ensure_knowledge_base_in_sync(chunks)
                self.bm25_search = BM25Search(chunks)

            self.vector_search = VectorSearch()
            self.hybrid_search = HybridSearch(
                self.vector_search,
                self.bm25_search,
                rrf_k=settings.rrf_k,
            )
            results = self.hybrid_search.search(
                query,
                top_k=max(settings.dense_top_k, settings.bm25_top_k),
                dense_top_k=settings.dense_top_k,
                bm25_top_k=settings.bm25_top_k,
            )

        if self.reranker is None:
            self.reranker = Reranker()
        reranked_results = self.reranker.rerank(
            query,
            results,
            top_k=settings.reranker_top_k,
        )

        if self.rag_pipeline is None:
            self.rag_pipeline = RAGPipeline()
        pipeline_result = self.rag_pipeline.run(query, reranked_results)

        return {
            "query": query,
            "results": reranked_results,
            **pipeline_result,
        }
