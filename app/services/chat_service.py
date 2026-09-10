"""Chat routing and orchestration service."""

import asyncio
import logging
import re
from collections.abc import Mapping, Sequence
from typing import Literal

from app.generation.deepseek import DeepSeekGenerator
from app.infrastructure.order_mcp_client import get_order_via_mcp
from app.services.mcp_service import MCPService
from app.services.rag_service import RAGService
from app.services.item_service import ItemService
from app.rag.pipeline import RAGPipeline
from app.retrieval.bm25 import BM25Search
from app.retrieval.hybrid_search import HybridSearch
from app.retrieval.reranker import Reranker
from app.retrieval.vector_search import VectorSearch
from config.settings import settings


logger = logging.getLogger(__name__)


Route = Literal["rag", "order", "rag_mcp", "xianyu", "missing_order_id", "unsupported_action"]
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
    "状态",
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
_COMBINED_RULE_TERMS = (
    "一般",
    "通常",
    "规则",
    "政策",
    "多久",
    "多长时间",
    "时效",
)
_XIANYU_PRICE_TERMS = ("价格", "多少钱", "标价", "售价", "多少元", "price", "cost")
_XIANYU_STATUS_TERMS = (
    "在吗",
    "还有吗",
    "在售",
    "卖出",
    "售出",
    "已售",
    "状态",
    "available",
    "sold",
)
_XIANYU_KNOWLEDGE_TERMS = (
    "配件",
    "包含",
    "附带",
    "成色",
    "瑕疵",
    "磕碰",
    "功能",
    "检测",
    "维修",
    "拆修",
    "改装",
    "使用",
    "续航",
    "发货",
    "运费",
    "售后",
    "包邮",
    "说明",
    "accessory",
    "condition",
    "shipping",
    "repair",
)
_XIANYU_FACT_QUERY_PATTERNS = (
    re.compile(r"价格(?:是|为)?多少(?:元)?", re.IGNORECASE),
    re.compile(r"(?:多少钱|标价|售价|多少元|price|cost)", re.IGNORECASE),
    re.compile(
        r"(?:这个商品|商品)?(?:现在|当前)?(?:还)?(?:在吗|有吗|有货吗|在售吗|还有吗)",
        re.IGNORECASE,
    ),
    re.compile(r"(?:售卖)?状态(?:如何|怎么样|是什么)?", re.IGNORECASE),
    re.compile(r"(?:卖出|售出|已售|available|sold)", re.IGNORECASE),
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
    """Identify an order question that also asks about a platform rule."""
    return order_id is not None and any(term in query for term in _COMBINED_RULE_TERMS)


def route_query(query: str) -> RouteResult:
    """Route a query to RAG, MCP, or the minimal combined workflow."""

    normalized_query = query.strip() if isinstance(query, str) else ""
    if _is_unsupported_action(normalized_query):
        return "unsupported_action", None

    order_id = _extract_order_id(normalized_query)
    if _is_combined_query(normalized_query, order_id):
        return "rag_mcp", order_id

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
        generator: DeepSeekGenerator | None = None,
        rag_service: RAGService | None = None,
        mcp_service: MCPService | None = None,
        xianyu_rag_service: RAGService | None = None,
        item_service: ItemService | None = None,
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
        self.mcp_service = mcp_service or MCPService(order_lookup=get_order_via_mcp)
        self.xianyu_rag_service = xianyu_rag_service
        self.item_service = item_service or ItemService()

        # Preserve the old dependency attributes for existing callers/tests.
        self.vector_search = self.rag_service.vector_search
        self.bm25_search = self.rag_service.bm25_search
        self.hybrid_search = self.rag_service.hybrid_search
        self.reranker = self.rag_service.reranker
        self.rag_pipeline = getattr(self.rag_service, "rag_pipeline", rag_pipeline)
        self.generator = generator or getattr(self.rag_service, "generator", None)

    async def chat_async(
        self,
        query: str,
        *,
        scenario: str = "ecommerce",
        item_id: str | None = None,
    ) -> dict[str, object]:
        """Route a request to RAG, MCP, or both services."""

        if scenario not in {"ecommerce", "xianyu"}:
            raise ValueError("scenario must be either ecommerce or xianyu")
        if scenario == "xianyu":
            return await self._chat_xianyu(query, item_id)
        route, order_id = route_query(query)
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

    async def _chat_xianyu(
        self,
        query: str,
        item_id: str | None,
    ) -> dict[str, object]:
        """Answer a demo Xianyu item question with MCP facts plus scoped RAG."""

        resolved_item_id = item_id or self.item_service.resolve_item_id(query)
        if not resolved_item_id:
            common_terms = (
                "发货", "运费", "售后", "规则", "多久", "包邮", "shipping", "shipping time"
            )
            if any(term in query.casefold() for term in common_terms):
                return await self._chat_xianyu_common(query)
            return self._xianyu_clarification(query)

        try:
            item = await self.mcp_service.get_item_info(resolved_item_id)
        except Exception:
            logger.exception("Item MCP lookup failed for item_id=%s", resolved_item_id)
            response = self._non_rag_response(
                query,
                "商品资料查询暂时失败，请稍后重试或转人工客服。",
                can_answer=False,
                route="xianyu",
                action="handoff",
                item_id=resolved_item_id,
            )
            response["next_step"] = "human_handoff"
            return response
        if not item.get("found"):
            return self._xianyu_clarification(query, item_id=resolved_item_id)

        lowered_query = query.casefold()
        asks_price = any(term in lowered_query for term in _XIANYU_PRICE_TERMS)
        asks_status = any(term in lowered_query for term in _XIANYU_STATUS_TERMS)
        asks_knowledge = any(term in lowered_query for term in _XIANYU_KNOWLEDGE_TERMS)
        fact_answers: list[str] = []

        if asks_price:
            cents = int(item["listed_price_cents"])
            fact_answers.append(
                f"“{item['title']}”的卖家维护标价为 ¥{cents / 100:.2f}。"
                f"资料更新时间：{item['updated_at']}。"
            )
        if asks_status:
            status = item["sale_status"]
            if status == "listed":
                fact_answers.append(
                    f"“{item['title']}”的卖家资料标记为在售；下单前建议再次联系卖家确认。"
                )
            elif status == "sold":
                fact_answers.append(f"“{item['title']}”的卖家资料标记为已售出。")
            else:
                return self._xianyu_handoff(
                    query,
                    self._join_xianyu_answers(
                        fact_answers,
                        "该商品当前状态在卖家资料中标记为未知，无法替你确认，请联系卖家核实。",
                    ),
                    item,
                )

        if fact_answers and not asks_knowledge:
            return self._xianyu_reply(
                query,
                self._join_xianyu_answers(fact_answers),
                item,
            )

        retrieval_query = self._xianyu_retrieval_query(
            query,
            remove_facts=bool(fact_answers),
            item_title=str(item["title"]),
        )
        rag_service = self._get_xianyu_rag_service()
        try:
            warm_up = getattr(rag_service, "warm_up", None)
            if callable(warm_up):
                warm_up()
            prepared = await asyncio.to_thread(
                rag_service.prepare,
                retrieval_query,
                item_id=str(item["item_id"]),
            )
        except Exception:
            logger.exception("Xianyu RAG preparation failed for item_id=%s", item["item_id"])
            return self._xianyu_handoff(
                query,
                self._join_xianyu_answers(
                    fact_answers,
                    "商品资料检索暂时失败，请转人工客服确认。",
                ),
                item,
            )
        context = prepared.get("context")
        if not prepared.get("can_answer") or not isinstance(context, Mapping):
            return self._xianyu_handoff(
                query,
                self._join_xianyu_answers(
                    fact_answers,
                    "现有商品资料不足以可靠回答这个问题，请转人工客服确认。",
                ),
                item,
                prepared,
            )
        context_text = str(context.get("context", "")).strip()
        if not context_text:
            return self._xianyu_handoff(
                query,
                self._join_xianyu_answers(
                    fact_answers,
                    "现有商品资料没有覆盖这个问题，请转人工客服确认。",
                ),
                item,
                prepared,
            )
        try:
            public_context = self._public_item_context(item)
            answer = self._get_generator().generate(
                query,
                public_context + "\n\n卖家规则与商品说明：\n" + context_text,
            )
            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError("empty Xianyu answer")
            answer = self._join_xianyu_answers(fact_answers, answer.strip())
        except Exception:
            logger.exception("Xianyu answer generation failed for item_id=%s", item["item_id"])
            return self._xianyu_handoff(
                query,
                self._join_xianyu_answers(
                    fact_answers,
                    "商品问题回答生成失败，请转人工客服。",
                ),
                item,
                prepared,
            )
        return {
            "query": query,
            "route": "xianyu",
            "action": "reply",
            "answer": answer,
            "sources": prepared.get("sources", []),
            "results": prepared.get("results", []),
            "reliability": prepared.get("reliability"),
            "next_step": None,
            "can_answer": True,
            "item_id": item["item_id"],
            "item_info": item,
        }

    async def _chat_xianyu_common(self, query: str) -> dict[str, object]:
        """Answer an item-independent question from common seller rules only."""

        try:
            rag_service = self._get_xianyu_rag_service()
            warm_up = getattr(rag_service, "warm_up", None)
            if callable(warm_up):
                warm_up()
            prepared = await asyncio.to_thread(rag_service.prepare, query)
            context = prepared.get("context")
            if not prepared.get("can_answer") or not isinstance(context, Mapping):
                return self._non_rag_response(
                    query,
                    "现有卖家通用规则不足以可靠回答这个问题，请转人工客服。",
                    can_answer=False,
                    route="xianyu",
                    action="handoff",
                ) | {"next_step": "human_handoff"}
            context_text = str(context.get("context", "")).strip()
            if not context_text:
                raise RuntimeError("empty common Xianyu context")
            answer = self._get_generator().generate(query, "卖家通用规则：\n" + context_text)
            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError("empty common Xianyu answer")
            return {
                "query": query,
                "route": "xianyu",
                "action": "reply",
                "answer": answer.strip(),
                "sources": prepared.get("sources", []),
                "results": prepared.get("results", []),
                "reliability": prepared.get("reliability"),
                "next_step": None,
                "can_answer": True,
            }
        except Exception:
            logger.exception("Common Xianyu RAG failed")
            response = self._non_rag_response(
                query,
                "现有卖家通用规则没有覆盖这个问题，请转人工客服。",
                can_answer=False,
                route="xianyu",
                action="handoff",
            )
            response["next_step"] = "human_handoff"
            return response

    def _get_xianyu_rag_service(self) -> RAGService:
        if self.xianyu_rag_service is None:
            self.xianyu_rag_service = RAGService(
                generator=self.generator,
                knowledge_base_path=settings.xianyu_knowledge_base_path,
                collection_name=settings.xianyu_qdrant_collection,
                manifest_path=settings.xianyu_ingestion_manifest_path,
                scoped_corpus=True,
            )
        return self.xianyu_rag_service

    @staticmethod
    def _public_item_context(item: Mapping[str, object]) -> str:
        cents = int(item["listed_price_cents"])
        return (
            "公开商品事实（以卖家资料快照为准）：\n"
            f"商品：{item['title']}（{item['item_id']}）\n"
            f"卖家维护标价：¥{cents / 100:.2f}\n"
            f"资料状态：{item['sale_status']}\n"
            f"资料更新时间：{item['updated_at']}\n"
            "资料性质：卖家人工维护快照，不是闲鱼平台实时查询。\n"
            "回答边界：未记录的信息不得推断为不存在，特别是维修史、隐藏瑕疵和未列出的配件。"
        )

    @staticmethod
    def _join_xianyu_answers(parts: Sequence[str], tail: str | None = None) -> str:
        """Join deterministic facts with a grounded knowledge answer."""

        answer_parts = [part.strip() for part in parts if part.strip()]
        if tail is not None and tail.strip():
            answer_parts.append(tail.strip())
        return "\n".join(answer_parts)

    @staticmethod
    def _xianyu_retrieval_query(
        query: str,
        *,
        remove_facts: bool,
        item_title: str,
    ) -> str:
        """Keep structured fact wording from weakening knowledge retrieval."""

        cleaned = query
        if remove_facts:
            for pattern in _XIANYU_FACT_QUERY_PATTERNS:
                cleaned = pattern.sub(" ", cleaned)
        cleaned = re.sub(r"(?:这个|这件)?商品", " ", cleaned)
        cleaned = re.sub(r"[\s，,。.!！?？、；;：:]+", " ", cleaned).strip()
        cleaned = re.sub(r"^(?:和|及|与)\s*", "", cleaned).strip()
        normalized_title = item_title.strip()
        if normalized_title and normalized_title.casefold() not in cleaned.casefold():
            cleaned = f"{normalized_title} {cleaned}".strip()
        return cleaned or query

    @staticmethod
    def _xianyu_reply(query: str, answer: str, item: Mapping[str, object]) -> dict[str, object]:
        return {
            "query": query,
            "route": "xianyu",
            "action": "reply",
            "answer": answer,
            "sources": [],
            "results": [],
            "reliability": None,
            "next_step": None,
            "can_answer": True,
            "item_id": item["item_id"],
            "item_info": dict(item),
        }

    @staticmethod
    def _xianyu_clarification(query: str, item_id: str | None = None) -> dict[str, object]:
        response = ChatService._non_rag_response(
            query,
            "请提供具体商品编号（例如 DEMO_ITEM_001），我再帮你查询对应商品。",
            can_answer=False,
            route="xianyu",
            action="clarify",
        )
        if item_id is not None:
            response["item_id"] = item_id
        response["next_step"] = "clarify_question"
        return response

    @staticmethod
    def _xianyu_handoff(
        query: str,
        answer: str,
        item: Mapping[str, object],
        prepared: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        response = ChatService._non_rag_response(
            query,
            answer,
            can_answer=False,
            route="xianyu",
            action="handoff",
        )
        response.update(
            {
                "item_id": item["item_id"],
                "item_info": dict(item),
                "next_step": "human_handoff",
            }
        )
        if prepared is not None:
            response.update(
                {
                    "sources": prepared.get("sources", []),
                    "results": prepared.get("results", []),
                    "reliability": prepared.get("reliability"),
                }
            )
        return response

    async def _chat_rag_mcp(
        self,
        query: str,
        order_id: str | None,
    ) -> dict[str, object]:
        """Run RAG and MCP concurrently, then make one final model call."""
        rag_result, mcp_result = await asyncio.gather(
            asyncio.to_thread(self.rag_service.prepare, query),
            self.mcp_service.get_order(order_id) if order_id else self._missing_order_result(),
            return_exceptions=True,
        )

        if isinstance(rag_result, Exception):
            logger.error("Combined RAG preparation failed: %s", rag_result)
            return self._non_rag_response(
                query,
                "本次综合查询失败，请稍后重试。",
                can_answer=False,
                route="rag_mcp",
            )
        if isinstance(mcp_result, Exception):
            logger.error("Combined MCP lookup failed for order_id=%s: %s", order_id, mcp_result)
            return self._non_rag_response(
                query,
                "本次订单查询失败，请稍后重试。",
                can_answer=False,
                route="rag_mcp",
            )
        if not mcp_result.get("found"):
            actual_order_id = mcp_result.get("order_id") or order_id
            return self._non_rag_response(
                query,
                f"未查询到模拟订单 {actual_order_id}，请核对订单号。",
                can_answer=True,
                route="rag_mcp",
            )

        context = rag_result.get("context")
        if not isinstance(context, dict) or not str(context.get("context", "")).strip():
            return self._non_rag_response(
                query,
                "知识库中没有足够的发货规则信息，请转人工客服。",
                can_answer=False,
                route="rag_mcp",
            )

        try:
            answer = self._get_generator().generate_combined(
                query,
                rag_result,
                mcp_result,
            )
            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError("DeepSeek returned an empty combined answer")
        except Exception:
            logger.exception("Combined answer generation failed")
            return self._non_rag_response(
                query,
                "本次综合查询回答失败，请稍后重试。",
                can_answer=False,
                route="rag_mcp",
            )

        return {
            "query": query,
            "route": "rag_mcp",
            "answer": answer.strip(),
            "sources": rag_result.get("sources", []),
            "results": rag_result.get("results", []),
            "reliability": rag_result.get("reliability"),
            "next_step": "complete",
            "can_answer": True,
            "rag_result": rag_result,
            "mcp_result": mcp_result,
        }

    @staticmethod
    async def _missing_order_result() -> dict[str, object]:
        return {"found": False, "order_id": None, "error": "missing_order_id"}

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

    @staticmethod
    def _non_rag_response(
        query: str,
        answer: str,
        *,
        can_answer: bool,
        route: str | None = None,
        action: str | None = None,
        item_id: str | None = None,
    ) -> dict[str, object]:
        response = {
            "query": query,
            "answer": answer,
            "sources": [],
            "results": [],
            "reliability": None,
            "next_step": None,
            "can_answer": can_answer,
        }
        if route is not None:
            response["route"] = route
        if action is not None:
            response["action"] = action
        if item_id is not None:
            response["item_id"] = item_id
        return response

    def chat(self, query: str) -> dict[str, object]:
        """Delegate the ordinary RAG path to RAGService."""
        return self.rag_service.chat(query)
