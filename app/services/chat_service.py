"""Chat routing and orchestration service."""

import asyncio
import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from app.generation.deepseek import DeepSeekGenerator
from app.infrastructure.order_mcp_client import get_order_via_mcp
from app.services.mcp_service import MCPService
from app.services.rag_service import RAGService
from app.services.item_service import ItemService
from app.services.session_manager import SessionManager
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
_XIANYU_ITEM_CONTEXT_TERMS = (
    "这个商品",
    "这件商品",
    "这个东西",
    "这件",
    "它",
    "配件",
    "成色",
    "瑕疵",
    "磕碰",
    "维修",
    "拆修",
    "改装",
    "标价",
    "售价",
    "多少钱",
    "在售",
    "售出",
    "卖出",
    "有货",
    "accessory",
    "condition",
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

    if order_id is not None and any(term in query for term in ("查", "查询", "查看")):
        return True

    mentions_order_detail = any(term in query for term in _ORDER_QUERY_TERMS)
    has_personal_order = any(term in query for term in _PERSONAL_ORDER_TERMS)
    return mentions_order_detail and (order_id is not None or has_personal_order)


def _session_order_id(session_state: Mapping[str, Any] | None) -> str | None:
    if not isinstance(session_state, Mapping):
        return None
    order_id = session_state.get("order_id")
    return str(order_id).strip().upper() if order_id else None


def _history_order_id(history: Sequence[Mapping[str, Any]] | None) -> str | None:
    if not history:
        return None
    for item in reversed(history):
        if not isinstance(item, Mapping):
            continue
        order_id = _extract_order_id(str(item.get("content", "")))
        if order_id:
            return order_id
    return None


def _is_contextual_followup(query: str, session_order_id: str | None) -> bool:
    return session_order_id is not None and any(
        term in query for term in ("它", "这个订单", "这笔", "那", "现在", "当前", "超时")
    )


def _is_combined_query(
    query: str,
    order_id: str | None,
    session_order_id: str | None,
) -> bool:
    if order_id is None:
        return False
    asks_for_rule = any(term in query for term in _COMBINED_RULE_TERMS)
    asks_for_current_order = any(
        term in query for term in ("现在", "当前", "状态", "已经", "是否", "超时")
    )
    return asks_for_rule and asks_for_current_order or (
        session_order_id is not None and "超时" in query
    )


def _is_rule_followup(query: str, remembered_order_id: str | None) -> bool:
    """Keep a policy-only follow-up on the RAG path."""
    return remembered_order_id is not None and any(
        term in query for term in _COMBINED_RULE_TERMS
    ) and not any(term in query for term in ("现在", "当前", "状态", "已经", "是否", "超时"))


def _requires_item_context(query: str) -> bool:
    """Return whether the question cannot be answered without a concrete item."""

    lowered_query = str(query or "").casefold()
    return any(term in lowered_query for term in _XIANYU_ITEM_CONTEXT_TERMS)


def route_query(
    query: str,
    history: Sequence[Mapping[str, Any]] | None = None,
    session_state: Mapping[str, Any] | None = None,
) -> RouteResult:
    """Route a query to RAG, MCP, or the minimal combined workflow."""

    normalized_query = query.strip() if isinstance(query, str) else ""
    if _is_unsupported_action(normalized_query):
        return "unsupported_action", None

    current_order_id = _extract_order_id(normalized_query)
    remembered_order_id = _session_order_id(session_state) or _history_order_id(history)
    order_id = current_order_id or (
        remembered_order_id
        if _is_contextual_followup(normalized_query, remembered_order_id)
        else None
    )

    if _is_combined_query(normalized_query, order_id, remembered_order_id):
        return "rag_mcp", order_id

    if current_order_id is None and _is_rule_followup(normalized_query, remembered_order_id):
        return "rag", None

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
        session_manager: SessionManager | None = None,
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
        self.session_manager = session_manager or SessionManager()
        self.xianyu_rag_service = xianyu_rag_service
        self.item_service = item_service or ItemService()

        self.rag_pipeline = getattr(self.rag_service, "rag_pipeline", rag_pipeline)
        self.generator = generator or getattr(self.rag_service, "generator", None)

    async def chat_async(
        self,
        query: str,
        chat_id: str | None = None,
        *,
        item_id: str | None = None,
    ) -> dict[str, object]:
        """Resolve session item context, then route to existing capabilities."""

        resolved_chat_id, session = self.session_manager.get_or_create(chat_id)
        history, session_state = self.session_manager.read_context(session)
        current_item_id = self.session_manager.get_current_item_id(resolved_chat_id)
        item, resolution_response = await self._resolve_current_item(
            query,
            item_id,
            current_item_id,
        )
        if resolution_response is not None:
            self.session_manager.append_turn(
                resolved_chat_id,
                query,
                str(resolution_response.get("answer") or ""),
                last_intent="item_clarification",
            )
            if chat_id is not None:
                resolution_response["chat_id"] = resolved_chat_id
            return resolution_response

        if item is not None:
            self.session_manager.set_current_item_id(
                resolved_chat_id,
                str(item["item_id"]),
            )

        route, order_id = route_query(query, history, session_state)
        if route == "rag":
            if item is not None:
                response = await self._chat_xianyu(
                    query,
                    str(item["item_id"]),
                    item=item,
                )
            elif _requires_item_context(query) and not _is_rule_followup(
                query,
                _session_order_id(session_state),
            ):
                response = self._xianyu_clarification(query)
            else:
                response = self.chat(query)
        elif route == "rag_mcp":
            response = await self._chat_rag_mcp(query, order_id, history)
        elif route == "missing_order_id":
            response = self._non_rag_response(
                query,
                "请提供订单号，并重新发送完整问题，例如：帮我查订单 TEST1001。",
                can_answer=False,
            )
        elif route == "unsupported_action":
            response = self._non_rag_response(
                query,
                "本版本暂不支持取消订单、退款或修改地址等操作，仅支持订单状态查询。",
                can_answer=False,
            )
        else:
            assert order_id is not None
            response = await self._chat_order(query, order_id)

        remembered_order_id = order_id or session_state.get("order_id")
        self.session_manager.append_turn(
            resolved_chat_id,
            query,
            str(response.get("answer") or ""),
            order_id=remembered_order_id,
            last_intent=self._intent_for_route(route),
        )
        if chat_id is not None:
            response["chat_id"] = resolved_chat_id
        return response

    async def _resolve_current_item(
        self,
        query: str,
        structured_item_id: str | None,
        current_item_id: str | None,
    ) -> tuple[Mapping[str, object] | None, dict[str, object] | None]:
        """Resolve and confirm one item without falling back after a bad switch."""

        explicit_item_id = str(structured_item_id or "").strip().upper() or None
        text_item_ids = self.item_service.resolve_item_ids(query)
        if len(text_item_ids) > 1:
            return None, self._item_conflict_response(
                query,
                "问题里出现了多个商品，请明确本次要咨询的一个商品编号。",
            )
        text_item_id = text_item_ids[0] if text_item_ids else None
        if (
            explicit_item_id is not None
            and text_item_id is not None
            and explicit_item_id != text_item_id
        ):
            return None, self._item_conflict_response(
                query,
                "请求中的商品编号与问题文字提到的商品不一致，请确认本次要咨询哪一件。",
            )

        candidate_id = explicit_item_id or text_item_id
        has_new_candidate = candidate_id is not None
        if candidate_id is None:
            candidate_id = current_item_id
        if candidate_id is None:
            return None, None

        try:
            item = await self.mcp_service.get_item_info(candidate_id)
        except Exception:
            logger.exception("Item MCP lookup failed for item_id=%s", candidate_id)
            response = self._non_rag_response(
                query,
                "商品资料查询暂时失败，请稍后重试或转人工客服。",
                can_answer=False,
                route="xianyu",
                action="handoff",
                item_id=candidate_id,
            )
            response["next_step"] = "human_handoff"
            return None, response
        if not item.get("found"):
            message = (
                f"未找到商品 {candidate_id}，请核对商品编号。"
                if has_new_candidate
                else "当前会话关联的商品已无法确认，请重新提供商品编号。"
            )
            return None, self._item_conflict_response(query, message, item_id=candidate_id)
        return item, None

    async def _chat_xianyu(
        self,
        query: str,
        item_id: str | None,
        *,
        item: Mapping[str, object] | None = None,
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

        if item is None:
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
    def _item_conflict_response(
        query: str,
        answer: str,
        *,
        item_id: str | None = None,
    ) -> dict[str, object]:
        response = ChatService._non_rag_response(
            query,
            answer,
            can_answer=False,
            route="xianyu",
            action="clarify",
            item_id=item_id,
        )
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
        order_id: str,
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, object]:
        """Run RAG and MCP concurrently, then make one final model call."""
        rag_result, mcp_result = await asyncio.gather(
            asyncio.to_thread(self.rag_service.prepare, query),
            self.mcp_service.get_order(order_id),
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
            generator = self._get_generator()
            if history:
                answer = generator.generate_combined(
                    query,
                    rag_result,
                    mcp_result,
                    history=history,
                )
            else:
                answer = generator.generate_combined(query, rag_result, mcp_result)
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
    def _intent_for_route(route: Route) -> str:
        if route == "order":
            return "order_query"
        if route == "rag_mcp":
            return "rag_mcp"
        if route == "rag":
            return "rag_query"
        return route

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
