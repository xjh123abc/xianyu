"""RAG application service used by normal and combined chat flows."""

from __future__ import annotations

import logging
from typing import Any

from app.generation.deepseek import DeepSeekGenerator
from collections.abc import Callable

from app.ingestion.loader import load_configured_knowledge_base, load_knowledge_base, load_xianyu_knowledge_base
from app.ingestion.pipeline import ensure_knowledge_base_in_sync
from app.rag.pipeline import RAGPipeline
from app.retrieval.bm25 import BM25Search
from app.retrieval.hybrid_search import HybridSearch
from app.retrieval.reranker import Reranker
from app.retrieval.vector_search import VectorSearch
from app.retrieval.vector_search import xianyu_common_filter, xianyu_item_filter
from config.settings import settings


logger = logging.getLogger(__name__)


class RAGService:
    """Own retrieval, reranking, reliability gating, and RAG generation."""

    def __init__(
        self,
        vector_search: VectorSearch | BM25Search | HybridSearch | None = None,
        bm25_search: BM25Search | None = None,
        reranker: Reranker | None = None,
        rag_pipeline: RAGPipeline | None = None,
        generator: DeepSeekGenerator | None = None,
        *,
        hybrid_search_cls: type[HybridSearch] = HybridSearch,
        vector_search_cls: type[VectorSearch] = VectorSearch,
        reranker_cls: type[Reranker] = Reranker,
        knowledge_base_path: str | None = None,
        collection_name: str | None = None,
        manifest_path: str | None = None,
        scoped_corpus: bool = False,
        corpus_id: str | None = None,
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
        self.generator = generator
        self._hybrid_search_cls = hybrid_search_cls
        self._vector_search_cls = vector_search_cls
        self._reranker_cls = reranker_cls
        self.knowledge_base_path = knowledge_base_path
        self.collection_name = collection_name
        self.manifest_path = manifest_path
        self.scoped_corpus = scoped_corpus
        self.corpus_id = corpus_id

    def warm_up(self) -> None:
        """Load configured local retrieval models before worker-thread inference."""

        if self.hybrid_search is None and self.vector_search is None:
            if self.bm25_search is None:
                self.bm25_search = BM25Search(self._load_chunks())
            self.vector_search = self._vector_search_cls(**self._vector_search_kwargs())
            self.hybrid_search = self._hybrid_search_cls(
                self.vector_search,
                self.bm25_search,
                rrf_k=settings.rrf_k,
            )
        elif (
            self.hybrid_search is None
            and self.vector_search is not None
            and self.bm25_search is not None
        ):
            self.hybrid_search = self._hybrid_search_cls(
                self.vector_search,
                self.bm25_search,
                rrf_k=settings.rrf_k,
            )
        if self.reranker is None:
            self.reranker = self._reranker_cls()
        self._get_pipeline()

    def prepare(self, query: str, *, item_id: str | None = None) -> dict[str, object]:
        """Retrieve and gate evidence without calling DeepSeek."""
        query_filter = (
            xianyu_item_filter(item_id)
            if self.scoped_corpus and item_id
            else xianyu_common_filter()
            if self.scoped_corpus
            else None
        )
        chunk_filter = self._chunk_filter(item_id) if self.scoped_corpus else None
        reranked_results = self._retrieve_and_rerank(
            query,
            query_filter=query_filter,
            chunk_filter=chunk_filter,
        )
        pipeline = self._get_pipeline()
        if hasattr(pipeline, "run_after_rerank"):
            prepared = pipeline.run_after_rerank(reranked_results)
        else:
            # Compatibility for small injected pipeline doubles.
            prepared = pipeline.run(query, reranked_results)
        return {
            "query": query,
            "results": reranked_results,
            **prepared,
        }

    def chat(self, query: str, *, item_id: str | None = None) -> dict[str, object]:
        """Run RAG and convert dependency failures into a stable handoff result."""
        try:
            prepared = self.prepare(query, item_id=item_id)
            if not prepared["can_answer"]:
                return prepared

            context = prepared["context"]
            if context is None or not str(context["context"]).strip():
                return {
                    **prepared,
                    "can_answer": False,
                    "next_step": "human_handoff",
                    "answer": None,
                    "sources": [],
                }

            answer = self._get_generator().generate(query, context["context"])
            return {
                **prepared,
                "next_step": "complete",
                "answer": answer,
                "sources": context["sources"],
            }
        except Exception:
            logger.exception("RAG request failed")
            return {
                "query": query,
                "results": [],
                "can_answer": False,
                "next_step": "human_handoff",
                "reliability": None,
                "context": None,
                "answer": "知识库服务暂时不可用，请稍后重试或转人工客服。",
                "sources": [],
            }

    def _retrieve_and_rerank(
        self,
        query: str,
        *,
        query_filter: object | None = None,
        chunk_filter: Callable[[object], bool] | None = None,
    ) -> list[dict[str, Any]]:
        search_kwargs = {
            "top_k": max(settings.dense_top_k, settings.bm25_top_k),
            "dense_top_k": settings.dense_top_k,
            "bm25_top_k": settings.bm25_top_k,
        }
        if self.hybrid_search is not None:
            if query_filter is None and chunk_filter is None:
                results = self.hybrid_search.search(query, **search_kwargs)
            else:
                results = self.hybrid_search.search(
                    query,
                    **search_kwargs,
                    query_filter=query_filter,
                    bm25_filter=chunk_filter,
                )
        elif self.vector_search is not None and self.bm25_search is not None:
            self.hybrid_search = self._hybrid_search_cls(
                self.vector_search,
                self.bm25_search,
                rrf_k=settings.rrf_k,
            )
            if query_filter is None and chunk_filter is None:
                results = self.hybrid_search.search(query, **search_kwargs)
            else:
                results = self.hybrid_search.search(
                    query,
                    **search_kwargs,
                    query_filter=query_filter,
                    bm25_filter=chunk_filter,
                )
        elif self.vector_search is not None:
            if query_filter is None:
                points = self.vector_search.search(query, top_k=settings.dense_top_k)
            else:
                points = self.vector_search.search(
                    query, top_k=settings.dense_top_k, query_filter=query_filter
                )
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
                chunks = self._load_chunks()
                self.bm25_search = BM25Search(chunks)

            self.vector_search = self._vector_search_cls(**self._vector_search_kwargs())
            self.hybrid_search = self._hybrid_search_cls(
                self.vector_search,
                self.bm25_search,
                rrf_k=settings.rrf_k,
            )
            if query_filter is None and chunk_filter is None:
                results = self.hybrid_search.search(query, **search_kwargs)
            else:
                results = self.hybrid_search.search(
                    query,
                    **search_kwargs,
                    query_filter=query_filter,
                    bm25_filter=chunk_filter,
                )

        if self.reranker is None:
            self.reranker = self._reranker_cls()
        return self.reranker.rerank(query, results, top_k=settings.reranker_top_k)

    def _load_chunks(self) -> list[object]:
        """Load and verify the exact corpus used by this RAG service."""
        if self.scoped_corpus:
            chunks = load_xianyu_knowledge_base()
        elif self.knowledge_base_path is not None:
            chunks = load_knowledge_base(self.knowledge_base_path)
        else:
            chunks = load_configured_knowledge_base()
        if self.manifest_path is not None:
            ensure_knowledge_base_in_sync(chunks, manifest_path=self.manifest_path)
        elif self.scoped_corpus:
            # Xianyu has its own collection and manifest; fail closed if it was
            # not ingested instead of silently searching a stale/empty index.
            ensure_knowledge_base_in_sync(
                chunks,
                manifest_path=settings.xianyu_ingestion_manifest_path,
            )
        else:
            ensure_knowledge_base_in_sync(chunks)
        return chunks

    def _vector_search_kwargs(self) -> dict[str, str]:
        """Build the retriever identity used by the corresponding ingestion flow."""

        kwargs: dict[str, str] = {}
        if self.collection_name is not None:
            kwargs["collection_name"] = self.collection_name
        if self.corpus_id is not None:
            kwargs["corpus_id"] = self.corpus_id
        return kwargs

    @staticmethod
    def _chunk_filter(item_id: str | None) -> Callable[[object], bool] | None:
        normalized_id = str(item_id or "").strip()

        def allowed(chunk: object) -> bool:
            return (
                getattr(chunk, "scope", None) == "common"
                or (
                    normalized_id
                    and getattr(chunk, "scope", None) == "item"
                    and getattr(chunk, "item_id", None) == normalized_id
                )
            )

        return allowed

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
