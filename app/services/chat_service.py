"""Chat session orchestration and route selection.

Domain decisions live in focused handlers. This service owns only the chat
session lifecycle, handler selection, and the stable response contract used by
the API and channel workers.
"""

from __future__ import annotations

import asyncio
import logging

from app.generation.deepseek import DeepSeekGenerator
from app.infrastructure.order_mcp_client import get_order_via_mcp
from app.rag.pipeline import RAGPipeline
from app.retrieval.bm25 import BM25Search
from app.retrieval.hybrid_search import HybridSearch
from app.retrieval.reranker import Reranker
from app.retrieval.vector_search import VectorSearch
from app.services.chat_contracts import ChatMessage, SessionContext, Task
from app.services.chat_response import non_rag_response
from app.services.intent_router import IntentRouter
from app.services.item_service import ItemService
from app.services.knowledge_service import KnowledgeService
from app.services.mcp_service import MCPService
from app.services.order_chat_handler import OrderChatHandler
from app.services.order_router import route_query as legacy_route_query
from app.services.planner import Planner, planner_state
from app.services.result_merger import ResultMerger
from app.services.query_planner import (
    common_knowledge_query,
    xianyu_context_updates,
)
from app.services.rag_service import RAGService
from app.services.session_manager import SessionManager
from app.services.task_executor import (
    OrderTaskHandler,
    ServiceTaskHandler,
    TaskExecutor,
    XianyuExpertTaskHandler,
)
from app.services.xianyu.item_context_resolver import ItemContextResolver
from app.services.xianyu.item_fact_responder import ItemFactResponder
from app.services.xianyu.expert_orchestrator import XianyuExpertOrchestrator
from app.services.xianyu.experts.price_agent import PriceAgent
from app.services.xianyu.knowledge_responder import XianyuKnowledgeResponder
from app.services.xianyu.responses import clarification, merge_partial_response
from config.settings import settings


logger = logging.getLogger(__name__)

# Public compatibility for older callers.  ChatService itself now receives
# order routing only through Planner.
route_query = legacy_route_query


class ChatService:
    """Coordinate one buyer turn without owning domain-specific decisions."""

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
        intent_router: IntentRouter | None = None,
        expert_orchestrator: XianyuExpertOrchestrator | None = None,
        planner: Planner | None = None,
        task_executor: TaskExecutor | None = None,
        knowledge_service: KnowledgeService | None = None,
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
        self.session_manager = session_manager or SessionManager(
            database_path=settings.session_database_path,
            ttl_seconds=settings.session_ttl_seconds,
            max_sessions=settings.session_max_count,
            lock_timeout_seconds=settings.session_lock_timeout_seconds,
        )
        self.xianyu_rag_service = xianyu_rag_service
        self.item_service = item_service or ItemService()
        self.intent_router = intent_router or IntentRouter(
            classifier=self._classify_xianyu_intent
        )
        self.rag_pipeline = getattr(self.rag_service, "rag_pipeline", rag_pipeline)
        self.generator = generator or getattr(self.rag_service, "generator", None)

        # Providers read current attributes at call time so existing callers and
        # tests can still replace an infrastructure instance after construction.
        self.order_handler = OrderChatHandler(
            rag_service=lambda: self.rag_service,
            mcp_service=lambda: self.mcp_service,
            generator=self._get_generator,
        )
        self.item_context_resolver = ItemContextResolver(
            self.item_service,
            lambda item_id: self.mcp_service.get_item_info(item_id),
        )
        self.planner = planner or Planner(
            intent_router=self.intent_router,
            requires_item_context=self.item_context_resolver.requires_item_context,
            may_contain_explicit_item_reference=(
                self.item_context_resolver.may_contain_explicit_item_reference
            ),
        )
        self.price_agent = PriceAgent()
        self.item_fact_responder = ItemFactResponder(price_agent=self.price_agent)
        self.knowledge_service = knowledge_service or KnowledgeService(
            self._get_xianyu_rag_service
        )
        self.xianyu_knowledge_responder = XianyuKnowledgeResponder(
            knowledge_service=self.knowledge_service,
            generator=self._get_generator,
            fact_responder=self.item_fact_responder,
            route_intent=lambda query: self.intent_router.route(query),
        )
        self.expert_orchestrator = expert_orchestrator or XianyuExpertOrchestrator(
            fact_responder=self.item_fact_responder,
            knowledge_responder=self.xianyu_knowledge_responder,
            intent_router=self.intent_router,
            generator=self._get_generator,
            price_agent=self.price_agent,
        )
        xianyu_handler = XianyuExpertTaskHandler(
            expert_orchestrator=self.expert_orchestrator,
            item_loader=lambda item_id: self.mcp_service.get_item_info(item_id),
        )
        self.task_executor = task_executor or TaskExecutor(
            {
                "product": xianyu_handler,
                "price": xianyu_handler,
                "service": ServiceTaskHandler(
                    expert_handler=xianyu_handler,
                    knowledge_responder=self.xianyu_knowledge_responder,
                ),
                "order": OrderTaskHandler(order_handler=self.order_handler),
            }
        )
        self.result_merger = ResultMerger()

    async def chat_async(
        self,
        query: str,
        chat_id: str | None = None,
        *,
        item_id: str | None = None,
    ) -> dict[str, object]:
        """Serialize one session while resolving and answering its next turn."""

        include_chat_id = chat_id is not None
        resolved_chat_id, _ = self.session_manager.get_or_create(chat_id)
        async with self.session_manager.session_lock(resolved_chat_id):
            response = await self._chat_async_locked(
                query,
                resolved_chat_id,
                item_id=item_id,
            )
        if not include_chat_id:
            response.pop("chat_id", None)
        return response

    async def _chat_async_locked(
        self,
        query: str,
        chat_id: str,
        *,
        item_id: str | None = None,
    ) -> dict[str, object]:
        """Run the stable load → plan → execute transition boundary."""

        context = self.session_manager.load(chat_id)
        tasks = self.planner.plan(query, context, item_id=item_id)
        return await self._execute_planned_turn(
            query,
            chat_id,
            context,
            tasks,
            item_id=item_id,
        )

    async def _execute_planned_turn(
        self,
        query: str,
        chat_id: str,
        context: SessionContext,
        tasks: list[Task],
        *,
        item_id: str | None = None,
    ) -> dict[str, object]:
        """Compatibility executor while specialist handlers are migrated."""

        resolved_chat_id = chat_id
        current_item_id = context.current_item_id
        state = planner_state(tasks)
        intent_match = state.intent_match
        plan = state.question_plan
        text_item_ids = (
            self.item_context_resolver.resolve_text_item_ids(query)
            if state.needs_item
            else []
        )
        if state.needs_item:
            item, resolution_response = await self.item_context_resolver.resolve(
                query,
                item_id,
                current_item_id,
                text_item_ids=text_item_ids,
                defer_to_order_context=state.defer_to_order_context,
            )
        else:
            item, resolution_response = None, None

        if resolution_response is not None:
            if any(
                question["scope"] == "common"
                for question in plan["knowledge_questions"]
            ):
                common_response = await self.xianyu_knowledge_responder.handle_common(
                    common_knowledge_query(plan, query)
                )
                common_response["query"] = query
                resolution_response = merge_partial_response(
                    resolution_response,
                    common_response,
                )
            self.session_manager.save_turn(
                resolved_chat_id,
                query,
                str(resolution_response.get("answer") or ""),
                legacy_intent="item_clarification",
            )
            if resolution_response.get("item_id") is not None:
                self.item_fact_responder.attach_intent_metadata(
                    resolution_response,
                    intent_match,
                    None,
                )
            if chat_id is not None:
                resolution_response["chat_id"] = resolved_chat_id
            return resolution_response

        route, order_id = state.route, state.order_id
        pending_xianyu_context_updates: dict[str, object] | None = None
        if route == "rag":
            if item is not None or state.use_xianyu_without_item:
                expert_context = dict(context.platform_context.get("xianyu", {}))
                if item is not None and current_item_id != str(item["item_id"]):
                    expert_context = {
                        "item_id": str(item["item_id"]),
                        "recent_price_topic": None,
                        "shipping_condition": None,
                    }
                execution_item_id = (
                    str(item["item_id"])
                    if item is not None
                    else item_id or context.current_item_id
                )
                execution_context = SessionContext(
                    history=context.history,
                    current_item_id=execution_item_id,
                    current_order_id=context.current_order_id,
                    last_task_type=context.last_task_type,
                    negotiation=context.negotiation,
                    platform_context={
                        **context.platform_context,
                        "xianyu": {
                            **expert_context,
                            "original_query": query,
                            "resolved_item": dict(item),
                        }
                        if item is not None
                        else {**expert_context, "original_query": query},
                    },
                )
                message = ChatMessage(
                    "xianyu",
                    "seller",
                    chat_id,
                    "buyer",
                    execution_item_id,
                    query,
                )
                results = await self.task_executor.execute(tasks, message, execution_context)
                response = self.result_merger.merge(query, tasks, results)
                if item is not None:
                    updates = xianyu_context_updates(query, expert_context)
                    if updates:
                        pending_xianyu_context_updates = updates
            elif state.is_rule_followup:
                response = await asyncio.to_thread(self.chat, query)
            elif any(
                question["scope"] == "item"
                for question in plan["knowledge_questions"]
            ):
                response = clarification(query)
                if any(
                    question["scope"] == "common"
                    for question in plan["knowledge_questions"]
                ):
                    common_response = await self.xianyu_knowledge_responder.handle_common(
                        common_knowledge_query(plan, query)
                    )
                    common_response["query"] = query
                    response = merge_partial_response(response, common_response)
            elif plan["knowledge_questions"] and (
                current_item_id is not None
                or bool(plan["item_fields"])
                or state.use_xianyu_without_item
            ):
                response = await self.xianyu_knowledge_responder.handle_common(
                    common_knowledge_query(plan, query)
                )
                response["query"] = query
                if plan["item_fields"]:
                    response = merge_partial_response(clarification(query), response)
            elif self.item_context_resolver.requires_item_context(query):
                response = clarification(query)
            else:
                response = await asyncio.to_thread(self.chat, query)
        elif [task.task_type for task in tasks] == ["order", "service"]:
            message = ChatMessage(
                "xianyu",
                "seller",
                chat_id,
                "buyer",
                item_id,
                query,
            )
            results = await self.task_executor.execute(tasks, message, context)
            response = self.result_merger.merge(query, tasks, results)
        elif route == "missing_order_id":
            response = non_rag_response(
                query,
                "请提供订单号，并重新发送完整问题，例如：帮我查订单 TEST1001。",
                can_answer=False,
            )
        elif route == "unsupported_action":
            response = non_rag_response(
                query,
                "本版本暂不支持取消订单、退款或修改地址等操作，仅支持订单状态查询。",
                can_answer=False,
            )
        else:
            assert order_id is not None
            response = await self.order_handler.order(query, order_id)

        if item is not None:
            self.item_fact_responder.attach_intent_metadata(
                response,
                intent_match,
                item,
            )
        remembered_order_id = order_id or context.current_order_id
        turn_kwargs: dict[str, object] = {
            "current_order_id": remembered_order_id,
            "legacy_intent": (
                intent_match.intent.lower()
                if intent_match.intent != "OTHER"
                else self._intent_for_route(route)
            ),
        }
        if item is not None:
            turn_kwargs["current_item_id"] = str(item["item_id"])
        if pending_xianyu_context_updates:
            turn_kwargs["xianyu_context_updates"] = pending_xianyu_context_updates
        self.session_manager.save_turn(
            resolved_chat_id,
            query,
            str(response.get("answer") or ""),
            **turn_kwargs,
        )
        if chat_id is not None:
            response["chat_id"] = resolved_chat_id
        return response

    def _get_xianyu_rag_service(self) -> RAGService:
        """Lazily build the isolated seller/item knowledge retrieval service."""

        if self.xianyu_rag_service is None:
            self.xianyu_rag_service = RAGService(
                generator=self.generator,
                knowledge_base_path=settings.xianyu_knowledge_base_path,
                collection_name=settings.xianyu_qdrant_collection,
                manifest_path=settings.xianyu_ingestion_manifest_path,
                scoped_corpus=True,
                corpus_id=settings.xianyu_corpus_id,
            )
        return self.xianyu_rag_service

    def _classify_xianyu_intent(self, query: str) -> str | None:
        """Use the configured generator only for rule misses and classification."""

        classifier = getattr(self._get_generator(), "classify_intent", None)
        if not callable(classifier):
            return None
        result = classifier(query)
        return result if isinstance(result, str) else None

    @staticmethod
    def _xianyu_greeting(query: str) -> dict[str, object]:
        return non_rag_response(
            query,
            "你好，想了解什么？",
            can_answer=True,
            route="xianyu",
            action="reply",
        )

    @staticmethod
    def _intent_for_route(route: str) -> str:
        if route == "order":
            return "order_query"
        if route == "rag":
            return "rag_query"
        return route

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

    def chat(self, query: str) -> dict[str, object]:
        """Delegate the ordinary RAG path to RAGService."""

        return self.rag_service.chat(query)
