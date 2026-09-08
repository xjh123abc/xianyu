"""Chat orchestration and route selection."""

import asyncio
import json
import logging
import re
from typing import Literal

from app.generation.deepseek import DeepSeekGenerator
from app.infrastructure.order_mcp_client import get_order_via_mcp
from app.services.mcp_service import MCPService
from app.services.rag_service import RAGService
from app.rag.pipeline import RAGPipeline
from app.retrieval.bm25 import BM25Search
from app.retrieval.hybrid_search import HybridSearch
from app.retrieval.reranker import Reranker
from app.retrieval.vector_search import VectorSearch
from config.settings import settings


logger = logging.getLogger(__name__)


Route = Literal["rag", "order", "rag_mcp", "missing_order_id", "unsupported_action"]
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
_COMBINED_POLICY_PATTERN = re.compile(
    r"(?:订单|物流|运单|发货).{0,20}(?:退货|退款|售后|政策|规则|条件)"
    r"|(?:退货|退款|售后|政策|规则|条件).{0,20}(?:订单|物流|运单|发货)"
    r"|(?:结合|同时|并且).{0,20}(?:订单|物流|政策|规则)"
)


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


def _is_combined_query(query: str, order_id: str | None) -> bool:
    """Recognize explicit requests that need both order data and policy text."""
    return order_id is not None and _COMBINED_POLICY_PATTERN.search(query) is not None


def route_query(query: str) -> RouteResult:
    """Route a query to RAG, MCP, or the explicit combined branch."""

    normalized_query = query.strip() if isinstance(query, str) else ""
    if _is_unsupported_action(normalized_query):
        return "unsupported_action", None

    order_id = _extract_order_id(normalized_query)
    configured_route = getattr(settings, "route", "auto") if settings is not None else "auto"
    if str(configured_route).strip().lower() == "rag_mcp":
        return "rag_mcp", order_id

    if _is_combined_query(normalized_query, order_id):
        return "rag_mcp", order_id

    if _is_order_query(normalized_query, order_id):
        if order_id is None:
            return "missing_order_id", None
        return "order", order_id

    return "rag", None


class ChatService:
    """Coordinate RAG and MCP services behind the single chat endpoint."""

    def __init__(
        self,
        vector_search: VectorSearch | BM25Search | HybridSearch | None = None,
        bm25_search: BM25Search | None = None,
        reranker: Reranker | None = None,
        rag_pipeline: RAGPipeline | None = None,
        generator: DeepSeekGenerator | None = None,
        rag_service: RAGService | None = None,
        mcp_service: MCPService | None = None,
    ) -> None:
        self.rag_service = rag_service or RAGService(
            vector_search=vector_search,
            bm25_search=bm25_search,
            reranker=reranker,
            rag_pipeline=rag_pipeline,
            generator=generator,
            hybrid_search_cls=HybridSearch,
            vector_search_cls=VectorSearch,
            reranker_cls=Reranker,
        )
        # Passing the module-level function keeps the existing monkeypatch seam
        # and allows tests or callers to inject another MCP transport.
        self.mcp_service = mcp_service or MCPService(order_lookup=get_order_via_mcp)

        # Compatibility attributes for callers that used the old ChatService
        # constructor as a thin container for retrieval dependencies.
        self.vector_search = self.rag_service.vector_search
        self.bm25_search = self.rag_service.bm25_search
        self.hybrid_search = self.rag_service.hybrid_search
        self.reranker = self.rag_service.reranker
        self.rag_pipeline = getattr(self.rag_service, "rag_pipeline", rag_pipeline)
        self.generator = generator or getattr(self.rag_service, "generator", None)

    async def chat_async(self, query: str) -> dict[str, object]:
        """Route a request and await only the services needed by that route."""

        route, order_id = route_query(query)
        logger.info("chat route=%s order_id=%s", route, order_id)
        if route == "rag":
            return self.chat(query)
        if route == "rag_mcp":
            return await self._chat_rag_mcp(query, order_id)
        if route == "missing_order_id":
            return self._non_rag_response(
                query,
                "请提供订单号，并重新发送完整问题，例如：帮我查订单 TEST1001。",
                can_answer=False,
            )
        if route == "unsupported_action":
            return self._non_rag_response(
                query,
                "本版本暂不支持取消订单、退款或修改地址等操作，仅支持订单状态查询。",
                can_answer=False,
            )

        if order_id is None:
            return self._non_rag_response(
                query,
                "请提供订单号，并重新发送完整问题，例如：帮我查订单 TEST1001。",
                can_answer=False,
            )
        return await self._chat_order(query, order_id)

    async def _chat_rag_mcp(
        self,
        query: str,
        order_id: str | None,
    ) -> dict[str, object]:
        """Run RAG preparation and MCP lookup concurrently, then synthesize."""
        rag_task = asyncio.to_thread(self.rag_service.prepare, query)
        mcp_task = self.mcp_service.run(order_id)
        rag_value, mcp_value = await asyncio.gather(
            rag_task,
            mcp_task,
            return_exceptions=True,
        )

        rag_result: dict[str, object]
        if isinstance(rag_value, Exception):
            logger.error("Combined RAG preparation failed: %s", rag_value)
            rag_result = {"query": query, "error": str(rag_value)}
        else:
            rag_result = rag_value

        if isinstance(mcp_value, Exception):
            logger.error(
                "Combined MCP lookup failed for order_id=%s: %s",
                order_id,
                mcp_value,
            )
            mcp_result: dict[str, object] = {
                "found": False,
                "order_id": order_id,
                "error": str(mcp_value),
            }
        else:
            mcp_result = mcp_value

        if mcp_result.get("error") or not mcp_result.get("found"):
            answer = (
                "请提供订单号后再查询。"
                if mcp_result.get("error") == "missing_order_id"
                else f"未查询到模拟订单 {mcp_result.get('order_id') or order_id}，请核对订单号。"
            )
            return {
                "query": query,
                "answer": answer,
                "sources": rag_result.get("sources", []),
                "results": rag_result.get("results", []),
                "reliability": rag_result.get("reliability"),
                "next_step": None,
                "can_answer": mcp_result.get("error") != "missing_order_id",
                "rag_result": rag_result,
                "mcp_result": mcp_result,
            }

        try:
            answer = self._generate_combined(query, rag_result, mcp_result)
        except Exception:
            logger.exception("Combined RAG/MCP answer generation failed")
            return {
                "query": query,
                "answer": "本次综合查询回答失败，请稍后重试。",
                "sources": rag_result.get("sources", []),
                "results": rag_result.get("results", []),
                "reliability": rag_result.get("reliability"),
                "next_step": None,
                "can_answer": False,
                "rag_result": rag_result,
                "mcp_result": mcp_result,
            }

        return {
            "query": query,
            "answer": answer,
            "sources": rag_result.get("sources", []),
            "results": rag_result.get("results", []),
            "reliability": rag_result.get("reliability"),
            "next_step": "complete",
            "can_answer": True,
            "rag_result": rag_result,
            "mcp_result": mcp_result,
        }

    async def _chat_order(self, query: str, order_id: str) -> dict[str, object]:
        try:
            order_data = await self.mcp_service.get_order(order_id)
        except Exception:
            logger.exception("Order MCP lookup failed for order_id=%s", order_id)
            return self._non_rag_response(
                query,
                "本次订单查询失败，请稍后重试。",
                can_answer=False,
            )

        if not order_data.get("found"):
            actual_order_id = order_data.get("order_id") or order_id
            return self._non_rag_response(
                query,
                f"未查询到模拟订单 {actual_order_id}，请核对订单号。",
                can_answer=True,
            )

        try:
            answer = self._get_generator().generate_order(query, order_data)
            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError("DeepSeek returned an empty order answer")
        except Exception:
            logger.exception("Order answer generation failed for order_id=%s", order_id)
            return self._non_rag_response(
                query,
                "本次订单查询回答失败，请稍后重试。",
                can_answer=False,
            )

        return self._non_rag_response(query, answer.strip(), can_answer=True)

    def _get_generator(self) -> DeepSeekGenerator:
        if self.generator is not None:
            return self.generator

        if self.rag_pipeline is not None:
            pipeline_generator = getattr(self.rag_pipeline, "generator", None)
            if pipeline_generator is not None:
                self.generator = pipeline_generator
                return self.generator

        self.generator = DeepSeekGenerator()
        return self.generator

    def _generate_combined(
        self,
        query: str,
        rag_result: dict[str, object],
        mcp_result: dict[str, object],
    ) -> str:
        generator = self._get_generator()
        combined_generator = getattr(generator, "generate_combined", None)
        if callable(combined_generator):
            return combined_generator(query, rag_result, mcp_result)

        # Backward-compatible seam for simple injected generators.
        combined_context = json.dumps(
            {"rag_result": rag_result, "mcp_result": mcp_result},
            ensure_ascii=False,
            default=str,
        )
        return generator.generate(query, combined_context)

    @staticmethod
    def _non_rag_response(
        query: str,
        answer: str,
        *,
        can_answer: bool,
    ) -> dict[str, object]:
        return {
            "query": query,
            "answer": answer,
            "sources": [],
            "results": [],
            "reliability": None,
            "next_step": None,
            "can_answer": can_answer,
        }

    def chat(self, query: str) -> dict[str, object]:
        """Delegate ordinary RAG requests to the focused RAG service."""
        return self.rag_service.chat(query)
