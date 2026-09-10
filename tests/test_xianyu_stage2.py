"""Stage 2 acceptance checks for the isolated Xianyu chat scenario."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

from fastapi.testclient import TestClient

from app.api import chat as chat_api
from app.ingestion.loader import load_xianyu_knowledge_base
from app.main import app
from app.retrieval.bm25 import BM25Search
from app.retrieval.vector_search import xianyu_common_filter, xianyu_item_filter
from app.services.chat_service import ChatService
from app.services.item_service import ItemService


class FakeRag:
    def __init__(self, *, can_answer: bool = True) -> None:
        self.can_answer = can_answer
        self.item_ids: list[str | None] = []
        self.queries: list[str] = []
        self.warm_up_calls = 0

    def warm_up(self) -> None:
        self.warm_up_calls += 1

    def prepare(self, query: str, *, item_id: str | None = None):
        self.queries.append(query)
        self.item_ids.append(item_id)
        return {
            "can_answer": self.can_answer,
            "context": {"context": "confirmed item description", "sources": []}
            if self.can_answer
            else None,
            "sources": [],
            "results": [],
            "reliability": None,
        }


def _service(rag: FakeRag | None = None) -> ChatService:
    mcp = Mock()
    mcp.get_item_info = AsyncMock(side_effect=lambda item_id: ItemService().get_item_info(item_id))
    generator = Mock()
    generator.generate.return_value = "基于商品资料的回答"
    return ChatService(
        mcp_service=mcp,
        xianyu_rag_service=rag or FakeRag(),
        generator=generator,
    )


def test_xianyu_loader_marks_common_and_item_scopes() -> None:
    chunks = load_xianyu_knowledge_base()
    assert {chunk.scope for chunk in chunks} == {"common", "item"}
    assert {chunk.item_id for chunk in chunks if chunk.scope == "item"} == {
        "DEMO_ITEM_001",
        "DEMO_ITEM_002",
        "DEMO_ITEM_003",
    }


def test_bm25_filter_is_applied_before_top_k() -> None:
    chunks = load_xianyu_knowledge_base()
    search = BM25Search(chunks)
    results = search.search(
        "配件",
        top_k=10,
        filter_fn=lambda chunk: chunk.scope == "common"
        or chunk.item_id == "DEMO_ITEM_001",
    )
    assert all(result.get("item_id") in {None, "DEMO_ITEM_001"} for result in results)


def test_bm25_does_not_leak_a_more_relevant_other_item() -> None:
    chunks = load_xianyu_knowledge_base()
    search = BM25Search(chunks)

    results = search.search(
        "空格键磕碰",
        top_k=10,
        filter_fn=lambda chunk: chunk.scope == "common"
        or chunk.item_id == "DEMO_ITEM_001",
    )

    assert results
    assert all(result.get("item_id") in {None, "DEMO_ITEM_001"} for result in results)
    assert all("空格键" not in result["content"] for result in results)


def test_xianyu_qdrant_filters_are_common_or_selected_item() -> None:
    selected = xianyu_item_filter("DEMO_ITEM_001")
    common = xianyu_common_filter()
    assert selected.should and common.must
    assert len(selected.should) == 2
    assert selected.should[0].key == "scope"
    assert selected.should[0].match.value == "common"
    selected_item = selected.should[1]
    assert selected_item.must
    assert [(condition.key, condition.match.value) for condition in selected_item.must] == [
        ("scope", "item"),
        ("item_id", "DEMO_ITEM_001"),
    ]


def test_price_uses_readonly_mcp_without_rag() -> None:
    rag = FakeRag()
    result = asyncio.run(
        _service(rag).chat_async(
            "这个商品价格是多少？",
            scenario="xianyu",
            item_id="DEMO_ITEM_001",
        )
    )
    assert result["action"] == "reply"
    assert "1280.00" in str(result["answer"])
    assert rag.item_ids == []


def test_combined_price_and_item_details_use_mcp_and_scoped_rag() -> None:
    rag = FakeRag()

    result = asyncio.run(
        _service(rag).chat_async(
            "这个商品多少钱，带哪些配件？",
            scenario="xianyu",
            item_id="DEMO_ITEM_001",
        )
    )

    assert result["action"] == "reply"
    assert "1280.00" in str(result["answer"])
    assert "基于商品资料的回答" in str(result["answer"])
    assert rag.item_ids == ["DEMO_ITEM_001"]
    assert rag.queries == ["复古胶片相机套装 带哪些配件"]
    assert rag.warm_up_calls == 1


def test_combined_price_and_unsupported_detail_keeps_fact_and_handoffs() -> None:
    rag = FakeRag(can_answer=False)

    result = asyncio.run(
        _service(rag).chat_async(
            "这个商品多少钱，有完整维修记录吗？",
            scenario="xianyu",
            item_id="DEMO_ITEM_001",
        )
    )

    assert result["action"] == "handoff"
    assert result["can_answer"] is False
    assert "1280.00" in str(result["answer"])
    assert rag.item_ids == ["DEMO_ITEM_001"]
    assert rag.queries == ["复古胶片相机套装 有完整维修记录吗"]


def test_combined_price_and_status_use_structured_facts_without_rag() -> None:
    rag = FakeRag()

    result = asyncio.run(
        _service(rag).chat_async(
            "这个商品多少钱，是否已经售出？",
            scenario="xianyu",
            item_id="DEMO_ITEM_002",
        )
    )

    assert result["action"] == "reply"
    assert "560.00" in str(result["answer"])
    assert "已售出" in str(result["answer"])
    assert rag.item_ids == []


def test_item_independent_question_uses_common_rules_only() -> None:
    rag = FakeRag()
    result = asyncio.run(_service(rag).chat_async("通常多久发货？", scenario="xianyu"))
    assert result["action"] == "reply"
    assert result.get("item_id") is None
    assert rag.item_ids == [None]


def test_unknown_item_is_clarified_and_items_do_not_cross_talk() -> None:
    rag = FakeRag()
    service = _service(rag)
    missing = asyncio.run(service.chat_async("这个东西多少钱？", scenario="xianyu"))
    assert missing["action"] == "clarify"

    first = asyncio.run(
        service.chat_async("配件有哪些？", scenario="xianyu", item_id="DEMO_ITEM_001")
    )
    second = asyncio.run(
        service.chat_async("配件有哪些？", scenario="xianyu", item_id="DEMO_ITEM_002")
    )
    assert first["item_id"] == "DEMO_ITEM_001"
    assert second["item_id"] == "DEMO_ITEM_002"
    assert rag.item_ids == ["DEMO_ITEM_001", "DEMO_ITEM_002"]
    assert rag.queries == [
        "复古胶片相机套装 配件有哪些",
        "八成新机械键盘 配件有哪些",
    ]


def test_insufficient_item_documents_handoff() -> None:
    result = asyncio.run(
        _service(FakeRag(can_answer=False)).chat_async(
            "维修历史是什么？",
            scenario="xianyu",
            item_id="DEMO_ITEM_001",
        )
    )
    assert result["action"] == "handoff"
    assert result["can_answer"] is False


def test_chat_api_accepts_xianyu_fields(monkeypatch) -> None:
    fake_service = Mock()
    fake_service.chat_async = AsyncMock(
        return_value={
            "query": "价格？",
            "answer": "¥1280.00",
            "route": "xianyu",
            "action": "reply",
            "can_answer": True,
            "next_step": "complete",
            "item_id": "DEMO_ITEM_001",
            "item_info": ItemService().get_item_info("DEMO_ITEM_001"),
        }
    )
    monkeypatch.setattr(chat_api, "chat_service", fake_service)
    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "query": "价格？",
                "scenario": "xianyu",
                "item_id": "DEMO_ITEM_001",
            },
        )
    assert response.status_code == 200
    assert response.json()["item_id"] == "DEMO_ITEM_001"
    fake_service.chat_async.assert_awaited_once_with(
        "价格？", scenario="xianyu", item_id="DEMO_ITEM_001"
    )
