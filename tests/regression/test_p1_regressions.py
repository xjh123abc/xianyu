"""Regression coverage for the P1 core-chain review findings."""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.api import chat as chat_api
from app.main import app
from app.services.chat_service import ChatService
from app.services.rag_service import RAGService
from app.services.session_manager import SessionManager


def test_ordinary_rag_does_not_block_the_event_loop() -> None:
    entered = threading.Event()
    release = threading.Event()
    rag_service = Mock()

    def blocking_chat(query: str) -> dict[str, object]:
        entered.set()
        release.wait(timeout=2)
        return {"query": query, "answer": "done", "results": []}

    rag_service.chat.side_effect = blocking_chat
    service = ChatService(
        rag_service=rag_service,
        session_manager=SessionManager(),
    )

    async def exercise() -> dict[str, object]:
        task = asyncio.create_task(service.chat_async("ordinary knowledge question", "p1"))
        await asyncio.wait_for(asyncio.to_thread(entered.wait), timeout=0.5)
        release.set()
        return await asyncio.wait_for(task, timeout=0.5)

    started_at = time.monotonic()
    result = asyncio.run(exercise())

    assert time.monotonic() - started_at < 1
    assert result["answer"] == "done"
    rag_service.chat.assert_called_once_with("ordinary knowledge question")


def test_rag_service_degrades_retrieval_failure_to_handoff() -> None:
    service = RAGService()
    service.prepare = Mock(side_effect=ConnectionError("qdrant unavailable"))

    result = service.chat("shipping policy")

    assert result["query"] == "shipping policy"
    assert result["can_answer"] is False
    assert result["next_step"] == "clarify"
    assert result["context"] is None
    assert result["sources"] == []
    assert result["results"] == []
    assert result["answer"]


def test_rag_service_degrades_generation_failure_to_handoff() -> None:
    service = RAGService(generator=Mock())
    service.prepare = Mock(
        return_value={
            "query": "shipping policy",
            "results": [{"content": "evidence"}],
            "can_answer": True,
            "next_step": "llm",
            "reliability": {"can_answer": True},
            "context": {
                "context": "evidence",
                "sources": [{"source": "policy.md", "index": 0}],
            },
            "answer": None,
            "sources": [{"source": "policy.md", "index": 0}],
        }
    )
    service.generator.generate.side_effect = TimeoutError("deepseek timeout")

    result = service.chat("shipping policy")

    assert result["can_answer"] is False
    assert result["next_step"] == "clarify"
    assert result["sources"] == []
    assert result["answer"]


def test_chat_api_returns_structured_handoff_when_rag_dependency_fails(
    monkeypatch,
) -> None:
    rag_service = RAGService()
    rag_service.prepare = Mock(side_effect=ConnectionError("qdrant unavailable"))
    monkeypatch.setattr(
        chat_api,
        "chat_service",
        ChatService(rag_service=rag_service, session_manager=SessionManager()),
    )

    response = TestClient(app).post(
        "/chat",
        json={"query": "shipping policy", "chat_id": "p1-api"},
    )

    assert response.status_code == 200
    assert response.json()["can_answer"] is False
    assert response.json()["next_step"] == "clarify"
    assert response.json()["sources"] == []
