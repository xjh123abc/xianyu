"""Regression checks for the thin chat orchestrator boundaries."""

from __future__ import annotations

import asyncio
from unittest.mock import Mock

from app.services.chat_service import ChatService
from app.services.intent_router import IntentRouter
from config.settings import settings


class _RegularRag:
    def chat(self, query: str) -> dict[str, object]:
        return {
            "query": query,
            "answer": "普通知识库回答",
            "sources": [],
            "results": [],
            "reliability": None,
            "next_step": "complete",
            "can_answer": True,
        }


def test_regular_rag_turn_does_not_read_xianyu_item_snapshot() -> None:
    """A broken item snapshot must not take down an unrelated RAG question."""

    item_service = Mock()
    item_service.resolve_item_ids.side_effect = AssertionError("should not read items")
    service = ChatService(
        rag_service=_RegularRag(),
        item_service=item_service,
        intent_router=IntentRouter(),
    )

    result = asyncio.run(service.chat_async("客服联系方式是什么？", "regular_rag"))

    assert result["answer"] == "普通知识库回答"
    item_service.resolve_item_ids.assert_not_called()


def test_xianyu_rag_service_uses_its_own_corpus_identity() -> None:
    service = ChatService(rag_service=_RegularRag(), intent_router=IntentRouter())

    xianyu_rag_service = service._get_xianyu_rag_service()

    assert xianyu_rag_service.corpus_id == settings.xianyu_corpus_id
