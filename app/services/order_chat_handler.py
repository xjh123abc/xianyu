"""Order MCP and combined RAG/MCP execution kept outside chat orchestration."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from app.generation.deepseek import DeepSeekGenerator
from app.services.chat_response import non_rag_response


logger = logging.getLogger(__name__)


class OrderChatHandler:
    """Execute order queries through the standalone MCP-backed path."""

    def __init__(
        self,
        *,
        mcp_service: Callable[[], Any],
        generator: Callable[[], DeepSeekGenerator],
    ) -> None:
        self._mcp_service = mcp_service
        self._generator = generator

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
