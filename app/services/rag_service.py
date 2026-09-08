"""Application service for retrieval-augmented answers."""

from __future__ import annotations

from typing import Any

from app.generation.deepseek import DeepSeekGenerator
from app.ingestion.loader import load_configured_knowledge_base
from app.rag.pipeline import RAGPipeline
from app.retrieval.bm25 import BM25Search
from app.retrieval.hybrid_search import HybridSearch
from app.retrieval.reranker import Reranker
from app.retrieval.vector_search import VectorSearch
from config.settings import settings


class RAGService:
    """Run retrieval and optionally generate a grounded answer.

    ``prepare`` deliberately stops before DeepSeek generation.  The combined
    route uses that method so it can join RAG evidence with MCP data and make
    exactly one final model call.
    """

    def __init__(
        self,
        vector_search: VectorSearch | BM25Search | HybridSearch | None = None,
        bm25_search: BM25Search | None = None,
        reranker: Reranker | None = None,
        rag_pipeline: RAGPipeline | None = None,
        generator: DeepSeekGenerator | None = None,
        hybrid_search_cls: type[HybridSearch] = HybridSearch,
        vector_search_cls: type[VectorSearch] = VectorSearch,
        reranker_cls: type[Reranker] = Reranker,
    ) -> None:
        if isinstance(vector_search, (BM25Search, HybridSearch)) and bm25_search is None:
            self.vector_search = None
            self.bm25_search = (
                vector_search if isinstance(vector_search, BM25Search) else None
            )
            self.hybrid_search = (
                vector_search if isinstance(vector_search, HybridSearch) else None
            )
        else:
            self.vector_search = vector_search
            self.bm25_search = bm25_search
            self.hybrid_search = None
        self.reranker = reranker
        self.rag_pipeline = rag_pipeline
        self.generator = generator
        self._hybrid_search_cls = hybrid_search_cls
        self._vector_search_cls = vector_search_cls
        self._reranker_cls = reranker_cls

    def prepare(self, query: str) -> dict[str, object]:
        """Retrieve, rerank, and apply the RAG reliability gate."""
        reranked_results = self._retrieve_and_rerank(query)
        pipeline = self._get_pipeline()

        # The fallback keeps small test doubles and older pipeline adapters
        # compatible.  The real RAGPipeline exposes run_after_rerank.
        if hasattr(pipeline, "run_after_rerank"):
            pipeline_result = pipeline.run_after_rerank(reranked_results)
        else:
            pipeline_result = pipeline.run(query, reranked_results)

        return {
            "query": query,
            "results": reranked_results,
            **pipeline_result,
        }

    def chat(self, query: str) -> dict[str, object]:
        """Return a normal RAG response, including a generated answer."""
        prepared = self.prepare(query)
        if not prepared.get("can_answer"):
            return prepared

        # A legacy test double may already have generated in ``run``.
        existing_answer = prepared.get("answer")
        if isinstance(existing_answer, str) and existing_answer.strip():
            return prepared

        context = prepared.get("context")
        if not isinstance(context, dict) or not str(context.get("context", "")).strip():
            return {
                **prepared,
                "can_answer": False,
                "next_step": "human_handoff",
                "answer": None,
                "sources": [],
            }

        answer = self._get_generator().generate(query, str(context["context"]))
        return {
            **prepared,
            "next_step": "complete",
            "answer": answer,
            "sources": context.get("sources", []),
        }

    def _retrieve_and_rerank(self, query: str) -> list[dict[str, Any]]:
        if self.hybrid_search is not None:
            results = self.hybrid_search.search(
                query,
                top_k=max(settings.dense_top_k, settings.bm25_top_k),
                dense_top_k=settings.dense_top_k,
                bm25_top_k=settings.bm25_top_k,
            )
        elif self.vector_search is not None and self.bm25_search is not None:
            self.hybrid_search = self._hybrid_search_cls(
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

            self.vector_search = self._vector_search_cls()
            self.hybrid_search = self._hybrid_search_cls(
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
            self.reranker = self._reranker_cls()
        return self.reranker.rerank(
            query,
            results,
            top_k=settings.reranker_top_k,
        )

    def _get_pipeline(self) -> RAGPipeline:
        if self.rag_pipeline is None:
            self.rag_pipeline = RAGPipeline(generator=self.generator)
        return self.rag_pipeline

    def _get_generator(self) -> DeepSeekGenerator:
        if self.generator is not None:
            return self.generator

        pipeline_generator = getattr(self._get_pipeline(), "generator", None)
        if pipeline_generator is not None:
            self.generator = pipeline_generator
            return self.generator

        self.generator = DeepSeekGenerator()
        return self.generator
