"""Minimal chat service implementation."""

import re
from typing import Literal

from app.ingestion.loader import load_configured_knowledge_base
from app.rag.pipeline import RAGPipeline
from app.retrieval.bm25 import BM25Search
from app.retrieval.hybrid_search import HybridSearch
from app.retrieval.reranker import Reranker
from app.retrieval.vector_search import VectorSearch
from config.settings import settings


Route = Literal["rag", "order", "missing_order_id", "unsupported_action"]
RouteResult = tuple[Route, str | None]

_ORDER_ID_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])TEST\d{4}(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_EXPLICIT_ORDER_LOOKUP_PATTERN = re.compile(
    r"(?:帮我|请|麻烦)?\s*"
    r"(?:查|查询|查看)\s*(?:一下|下)?\s*"
    r"(?:我的|这笔|这个)?\s*(?:订单|物流|运单|快递)"
)
_UNSUPPORTED_ACTION_PATTERNS = (
    re.compile(
        r"(?:帮我|请|我要|我想|想要|申请|办理|发起|执行|直接)?\s*"
        r"(?:取消|撤销|删除|关闭)(?:一下)?\s*订单"
    ),
    re.compile(
        r"(?:帮我|请|我要|我想|想要|申请|办理|发起|执行|直接)\s*"
        r"(?:申请)?\s*(?:退款|退货|退货退款)"
    ),
    re.compile(
        r"(?:帮我|请|我要|我想|想要|申请|办理|发起|执行|直接)?\s*"
        r"(?:修改|更改|变更|更换|改)(?:收货)?地址"
    ),
)
_ORDER_QUERY_TERMS = (
    "订单状态",
    "订单信息",
    "订单详情",
    "物流",
    "运单",
    "快递",
    "发货",
    "签收",
    "到货",
)
_PERSONAL_ORDER_TERMS = ("我的订单", "这笔订单", "这个订单", "这单", "我的物流")


def _extract_order_id(query: str) -> str | None:
    match = _ORDER_ID_PATTERN.search(query)
    return match.group(0).upper() if match else None


def _is_unsupported_action(query: str) -> bool:
    return any(pattern.search(query) for pattern in _UNSUPPORTED_ACTION_PATTERNS)


def _is_order_query(query: str, order_id: str | None) -> bool:
    if _EXPLICIT_ORDER_LOOKUP_PATTERN.search(query):
        return True

    mentions_order_detail = any(term in query for term in _ORDER_QUERY_TERMS)
    has_personal_order = any(term in query for term in _PERSONAL_ORDER_TERMS)
    return mentions_order_detail and (order_id is not None or has_personal_order)


def route_query(query: str) -> RouteResult:
    """Route a query to RAG or the narrowly scoped order MCP branch."""

    normalized_query = query.strip() if isinstance(query, str) else ""
    if _is_unsupported_action(normalized_query):
        return "unsupported_action", None

    order_id = _extract_order_id(normalized_query)
    if _is_order_query(normalized_query, order_id):
        if order_id is None:
            return "missing_order_id", None
        return "order", order_id

    return "rag", None


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
