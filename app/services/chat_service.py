"""Chat session orchestration and route selection.

Domain decisions live in focused handlers. This service owns only the chat
session lifecycle, handler selection, and the stable response contract used by
the API and channel workers.
"""

from __future__ import annotations

from collections.abc import Mapping

from app.generation.deepseek import DeepSeekGenerator
from app.infrastructure.order_mcp_client import get_order_via_mcp
from app.rag.pipeline import RAGPipeline
from app.retrieval.bm25 import BM25Search
from app.retrieval.hybrid_search import HybridSearch
from app.retrieval.reranker import Reranker
from app.retrieval.vector_search import VectorSearch
from app.services.chat_contracts import ChatMessage, SessionContext, Task
from app.services.intent_router import IntentRouter
from app.services.item_service import ItemService
from app.services.knowledge_service import KnowledgeService
from app.services.mcp_service import MCPService
from app.services.order_chat_handler import OrderChatHandler
from app.services.planner import Planner, planner_state
from app.services.result_merger import ResultMerger
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
from config.settings import settings


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
            legacy_item_handler=lambda query, item, history: (
                self.expert_orchestrator.handle(
                    query,
                    item=item,
                    history=history,
                )
            ),
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
                    general_rag=lambda: self.rag_service,
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
        """Execute Planner tasks through the single unified path."""

        state = planner_state(tasks)
        item: Mapping[str, object] | None = None
        resolution_response: dict[str, object] | None = None
        # Planner alone decides whether this is a RAG/item turn.  The resolver
        # only validates the selected item; it no longer performs routing.
        if state.requires_item_resolution:
            text_item_ids = self.item_context_resolver.resolve_text_item_ids(query)
            item, resolution_response = await self.item_context_resolver.resolve(
                query,
                item_id,
                context.current_item_id,
                text_item_ids=text_item_ids,
                defer_to_order_context=state.defer_to_order_context,
            )
        expert_context = self._expert_context(context, item)
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
                    **({"resolved_item": dict(item)} if item is not None else {}),
                },
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
        precomputed = self._item_resolution_results(tasks, resolution_response)
        if precomputed:
            results = await self.task_executor.execute(
                tasks,
                message,
                execution_context,
                precomputed_responses=precomputed,
            )
        else:
            results = await self.task_executor.execute(tasks, message, execution_context)
        response = self.result_merger.merge(query, tasks, results)

        if item is not None:
            self.item_fact_responder.attach_intent_metadata(
                response,
                state.intent_match,
                item,
            )
        turn_kwargs: dict[str, object] = {
            "current_order_id": state.order_id or context.current_order_id,
            "last_task_type": tasks[-1].task_type,
        }
        if item is not None:
            turn_kwargs["current_item_id"] = str(item["item_id"])
            updates = self.planner.xianyu_context_updates(query, expert_context)
            if updates:
                turn_kwargs["xianyu_context_updates"] = updates
        self.session_manager.save_turn(
            chat_id,
            query,
            str(response.get("answer") or ""),
            **turn_kwargs,
        )
        response["chat_id"] = chat_id
        return response

    @staticmethod
    def _item_resolution_results(
        tasks: list[Task],
        resolution_response: dict[str, object] | None,
    ) -> dict[str, dict[str, object]]:
        """Feed item clarification/conflict through ResultMerger as TaskResults."""

        if resolution_response is None:
            return {}
        item_tasks = [task for task in tasks if task.task_type in {"product", "price"}]
        if not item_tasks and all(task.task_type == "service" for task in tasks):
            item_tasks = list(tasks)
        return {task.task_id: resolution_response for task in item_tasks}

    @staticmethod
    def _expert_context(
        context: SessionContext,
        item: Mapping[str, object] | None,
    ) -> dict[str, object]:
        """Keep per-item session state isolated when an item switch is confirmed."""

        expert_context = dict(context.platform_context.get("xianyu", {}))
        if item is not None and context.current_item_id != str(item["item_id"]):
            return {
                "item_id": str(item["item_id"]),
                "recent_price_topic": None,
                "shipping_condition": None,
            }
        return expert_context

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
