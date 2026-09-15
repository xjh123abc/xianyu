"""Order MCP and combined RAG/MCP execution kept outside chat orchestration."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from app.generation.deepseek import DeepSeekGenerator
from app.services.chat_response import non_rag_response


logger = logging.getLogger(__name__)


class OrderChatHandler:
    """Execute order-only and order-plus-policy queries with injected services."""

    def __init__(
        self,
        *,
        rag_service: Callable[[], Any],
        mcp_service: Callable[[], Any],
        generator: Callable[[], DeepSeekGenerator],
    ) -> None:
        self._rag_service = rag_service
        self._mcp_service = mcp_service
        self._generator = generator

    async def combined(
        self,
        query: str,
        order_id: str,
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, object]:
        """Run independent retrieval and order lookups before one final generation."""

        rag_service = self._rag_service()
        mcp_service = self._mcp_service()
        rag_result, mcp_result = await asyncio.gather(
            asyncio.to_thread(rag_service.prepare, query),
            mcp_service.get_order(order_id),
            return_exceptions=True,
        )
        if isinstance(rag_result, Exception):
            logger.error("Combined RAG preparation failed: %s", rag_result)
            return non_rag_response(
                query, "本次综合查询失败，请稍后重试。", can_answer=False, route="rag_mcp"
            )
        if isinstance(mcp_result, Exception):
            logger.error("Combined MCP lookup failed for order_id=%s: %s", order_id, mcp_result)
            return non_rag_response(
                query, "本次订单查询失败，请稍后重试。", can_answer=False, route="rag_mcp"
            )
        if not mcp_result.get("found"):
            actual_order_id = mcp_result.get("order_id") or order_id
            return non_rag_response(
                query,
                f"未查询到模拟订单 {actual_order_id}，请核对订单号。",
                can_answer=True,
                route="rag_mcp",
            )

        context = rag_result.get("context")
        if not isinstance(context, dict) or not str(context.get("context", "")).strip():
            return non_rag_response(
                query,
                "知识库中没有足够的发货规则信息，请转人工客服。",
                can_answer=False,
                route="rag_mcp",
            )
        try:
            generator = self._generator()
            answer = (
                generator.generate_combined(query, rag_result, mcp_result, history=history)
                if history
                else generator.generate_combined(query, rag_result, mcp_result)
            )
            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError("DeepSeek returned an empty combined answer")
        except Exception:
            logger.exception("Combined answer generation failed")
            return non_rag_response(
                query, "本次综合查询回答失败，请稍后重试。", can_answer=False, route="rag_mcp"
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

    async def order(self, query: str, order_id: str) -> dict[str, object]:
        """Resolve and answer one order question."""

        try:
            order_data = await self._mcp_service().get_order(order_id)
        except Exception:
            logger.exception("Order MCP lookup failed for order_id=%s", order_id)
            return non_rag_response(query, "本次订单查询失败，请稍后重试。", can_answer=False)
        if not order_data.get("found"):
            actual_order_id = order_data.get("order_id") or order_id
            return non_rag_response(
                query, f"未查询到模拟订单 {actual_order_id}，请核对订单号。", can_answer=True
            )
        try:
            answer = self._generator().generate_order(query, order_data)
            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError("DeepSeek returned an empty order answer")
        except Exception:
            logger.exception("Order answer generation failed for order_id=%s", order_id)
            return non_rag_response(query, "本次订单查询回答失败，请稍后重试。", can_answer=False)
        return non_rag_response(query, answer.strip(), can_answer=True)
