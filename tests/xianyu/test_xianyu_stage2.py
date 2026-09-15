"""Checks for item facts, scoped knowledge, and unified item context."""

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
            "context": {
                "context": "confirmed item description",
                "sources": [{"source": "seller_rules.md", "index": 0}],
            }
            if self.can_answer
            else None,
            "sources": [{"source": "seller_rules.md", "index": 0}]
            if self.can_answer
            else [],
            "results": [],
            "reliability": None,
        }


def _service(rag: FakeRag | None = None) -> ChatService:
    mcp = Mock()
    mcp.get_item_info = AsyncMock(side_effect=lambda item_id: ItemService().get_item_info(item_id))
    generator = Mock()
    generator.generate_xianyu.return_value = "基于商品资料的回答"
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
            item_id="DEMO_ITEM_001",
        )
    )
    assert result["action"] == "reply"
    assert result["answer"] == "这件标价是 ¥1280.00。"
    assert rag.item_ids == []


def test_combined_price_and_missing_item_details_handoffs() -> None:
    rag = FakeRag()

    result = asyncio.run(
        _service(rag).chat_async(
            "这个商品多少钱，带哪些配件？",
            item_id="DEMO_ITEM_001",
        )
    )

    assert result["action"] == "handoff"
    assert result["answer"] == "稍等我看看"
    assert rag.item_ids == []
    assert rag.warm_up_calls == 0


def test_combined_price_and_unsupported_detail_keeps_fact_and_handoffs() -> None:
    rag = FakeRag(can_answer=False)

    result = asyncio.run(
        _service(rag).chat_async(
            "这个商品多少钱，有完整维修记录吗？",
            item_id="DEMO_ITEM_001",
        )
    )

    assert result["action"] == "handoff"
    assert result["can_answer"] is False
    assert result["answer"] == "稍等我看看"
    assert rag.item_ids == []


def test_combined_price_and_status_use_structured_facts_without_rag() -> None:
    rag = FakeRag()
    service = _service(rag)
    service.generator.generate_xianyu.return_value = "这件标价是 ¥560.00，已经出掉了。"

    result = asyncio.run(
        service.chat_async(
            "这个商品多少钱，是否已经售出？",
            item_id="DEMO_ITEM_002",
        )
    )

    assert result["action"] == "reply"
    assert "560.00" in str(result["answer"])
    assert "出掉了" in str(result["answer"])
    assert rag.item_ids == []
    assert service.generator.generate_xianyu.call_args.args[0] == "这个商品多少钱，是否已经售出？"


def test_item_independent_question_uses_existing_rag_without_item_lookup() -> None:
    scoped_rag = FakeRag()
    service = _service(scoped_rag)
    ordinary_rag = Mock()
    ordinary_rag.chat.return_value = {
        "query": "通常多久发货？",
        "answer": "按现有通用规则回答",
        "results": [],
    }
    service.rag_service = ordinary_rag

    result = asyncio.run(service.chat_async("通常多久发货？"))

    assert result["answer"] == "按现有通用规则回答"
    assert result.get("item_id") is None
    ordinary_rag.chat.assert_called_once_with("通常多久发货？")
    assert scoped_rag.item_ids == []
    service.mcp_service.get_item_info.assert_not_awaited()


def test_unknown_item_handoffs_and_items_do_not_cross_talk() -> None:
    rag = FakeRag()
    service = _service(rag)
    missing = asyncio.run(service.chat_async("这个东西多少钱？"))
    assert missing["action"] == "handoff"
    assert missing["answer"] == "稍等我看看"

    first = asyncio.run(
        service.chat_async("配件有哪些？", item_id="DEMO_ITEM_001")
    )
    second = asyncio.run(
        service.chat_async("配件有哪些？", item_id="DEMO_ITEM_002")
    )
    assert first["item_id"] == "DEMO_ITEM_001"
    assert second["item_id"] == "DEMO_ITEM_002"
    assert first["action"] == second["action"] == "reply"
    assert "相机机身" in str(first["answer"])
    assert "键盘本体" in str(second["answer"])
    assert rag.item_ids == []


def test_insufficient_item_documents_handoff() -> None:
    result = asyncio.run(
        _service(FakeRag(can_answer=False)).chat_async(
            "维修历史是什么？",
            item_id="DEMO_ITEM_001",
        )
    )
    assert result["action"] == "handoff"
    assert result["can_answer"] is False


def test_chat_api_accepts_unified_item_fields_without_mode(monkeypatch) -> None:
    fake_service = Mock()
    fake_service.chat_async = AsyncMock(
        return_value={
            "query": "价格？",
            "answer": "¥128000.00",
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
                "chat_id": "buyer_chat_001",
                "item_id": "DEMO_ITEM_001",
            },
        )
    assert response.status_code == 200
    assert response.json()["item_id"] == "DEMO_ITEM_001"
    fake_service.chat_async.assert_awaited_once_with(
        "价格？", "buyer_chat_001", item_id="DEMO_ITEM_001"
    )


def test_chat_openapi_exposes_only_unified_request_fields() -> None:
    request_schema = app.openapi()["components"]["schemas"]["ChatRequest"]

    assert {"query", "chat_id", "item_id"} <= request_schema["properties"].keys()
    assert {"query", "chat_id"} <= set(request_schema["required"])
    assert "scenario" not in request_schema["properties"]
    assert request_schema["additionalProperties"] is False


def test_removed_legacy_mode_is_rejected_at_api_boundary(monkeypatch) -> None:
    fake_service = Mock()
    fake_service.chat_async = AsyncMock(return_value={"query": "价格？", "results": []})
    monkeypatch.setattr(chat_api, "chat_service", fake_service)

    response = TestClient(app).post(
        "/chat",
        json={
            "query": "价格？",
            "chat_id": "legacy_chat_001",
            "scenario": "xianyu",
            "item_id": "DEMO_ITEM_001",
        },
    )

    assert response.status_code == 422
    fake_service.chat_async.assert_not_awaited()


def test_text_item_id_is_confirmed_and_saved_for_same_chat_followup() -> None:
    service = _service()

    first = asyncio.run(
        service.chat_async("DEMO_ITEM_001 的价格是多少？", "buyer_chat_001")
    )
    second = asyncio.run(service.chat_async("那它多少钱？", "buyer_chat_001"))

    assert first["item_id"] == second["item_id"] == "DEMO_ITEM_001"
    assert first["chat_id"] == second["chat_id"] == "buyer_chat_001"
    assert service.mcp_service.get_item_info.await_count == 2
    assert [call.args[0] for call in service.mcp_service.get_item_info.await_args_list] == [
        "DEMO_ITEM_001",
        "DEMO_ITEM_001",
    ]


def test_known_item_with_missing_fact_handoffs_without_internal_id() -> None:
    result = asyncio.run(
        _service().chat_async(
            "这是 50mm 镜头吗？",
            "buyer_chat_known_item_unknown_attribute",
            item_id="DEMO_ITEM_001",
        )
    )

    assert result["action"] == "handoff"
    assert result["item_id"] == "DEMO_ITEM_001"
    assert result["next_step"] == "human_handoff"
    assert result["answer"] == "稍等我看看"
    assert "商品编号" not in str(result["answer"])
    assert "DEMO_ITEM_001" not in str(result["answer"])


def test_vague_question_about_known_item_handoffs() -> None:
    result = asyncio.run(
        _service().chat_async(
            "这个怎么样？",
            "buyer_chat_known_item_vague_question",
            item_id="DEMO_ITEM_001",
        )
    )

    assert result["action"] == "handoff"
    assert result["next_step"] == "human_handoff"
    assert result["answer"] == "稍等我看看"


def test_chat_api_restores_current_item_for_same_chat(monkeypatch) -> None:
    service = _service()
    monkeypatch.setattr(chat_api, "chat_service", service)

    with TestClient(app) as client:
        first = client.post(
            "/chat",
            json={
                "query": "这个多少钱？",
                "chat_id": "test_memory_001",
                "item_id": "DEMO_ITEM_001",
            },
        )
        second = client.post(
            "/chat",
            json={"query": "那它多少钱？", "chat_id": "test_memory_001"},
        )

    assert first.status_code == second.status_code == 200
    assert second.json()["item_id"] == "DEMO_ITEM_001"
    assert service.session_manager.get_current_item_id("test_memory_001") == "DEMO_ITEM_001"


def test_chat_api_new_item_overwrites_current_item_for_same_chat(monkeypatch) -> None:
    service = _service()
    monkeypatch.setattr(chat_api, "chat_service", service)

    with TestClient(app) as client:
        client.post(
            "/chat",
            json={
                "query": "这个多少钱？",
                "chat_id": "test_memory_switch",
                "item_id": "DEMO_ITEM_001",
            },
        )
        client.post(
            "/chat",
            json={
                "query": "这个多少钱？",
                "chat_id": "test_memory_switch",
                "item_id": "DEMO_ITEM_002",
            },
        )
        followup = client.post(
            "/chat",
            json={"query": "那它多少钱？", "chat_id": "test_memory_switch"},
        )

    assert followup.status_code == 200
    assert followup.json()["item_id"] == "DEMO_ITEM_002"
    assert service.session_manager.get_current_item_id("test_memory_switch") == "DEMO_ITEM_002"


def test_chat_api_current_items_are_isolated_by_chat_id(monkeypatch) -> None:
    service = _service()
    monkeypatch.setattr(chat_api, "chat_service", service)

    with TestClient(app) as client:
        client.post(
            "/chat",
            json={
                "query": "这个多少钱？",
                "chat_id": "buyer_A",
                "item_id": "DEMO_ITEM_001",
            },
        )
        client.post(
            "/chat",
            json={
                "query": "这个多少钱？",
                "chat_id": "buyer_B",
                "item_id": "DEMO_ITEM_002",
            },
        )
        buyer_a = client.post(
            "/chat",
            json={"query": "那它多少钱？", "chat_id": "buyer_A"},
        )
        buyer_b = client.post(
            "/chat",
            json={"query": "那它多少钱？", "chat_id": "buyer_B"},
        )

    assert buyer_a.status_code == buyer_b.status_code == 200
    assert buyer_a.json()["item_id"] == "DEMO_ITEM_001"
    assert buyer_b.json()["item_id"] == "DEMO_ITEM_002"


def test_structured_and_text_item_conflict_handoffs() -> None:
    service = _service()

    result = asyncio.run(
        service.chat_async(
            "我问的是 DEMO_ITEM_002",
            "buyer_chat_conflict",
            item_id="DEMO_ITEM_001",
        )
    )

    assert result["action"] == "handoff"
    assert result["can_answer"] is False
    assert result["answer"] == "稍等我看看"
    service.mcp_service.get_item_info.assert_not_awaited()


def test_multiple_text_items_handoff() -> None:
    service = _service()

    result = asyncio.run(
        service.chat_async(
            "比较 DEMO_ITEM_001 和 DEMO_ITEM_002 的价格",
            "buyer_chat_multiple",
        )
    )

    assert result["action"] == "handoff"
    assert result["answer"] == "稍等我看看"
    service.mcp_service.get_item_info.assert_not_awaited()


def test_valid_item_switch_replaces_old_item_without_cross_session_leak() -> None:
    service = _service()

    asyncio.run(
        service.chat_async("多少钱？", "buyer_chat_a", item_id="DEMO_ITEM_001")
    )
    switched = asyncio.run(
        service.chat_async("多少钱？", "buyer_chat_a", item_id="DEMO_ITEM_002")
    )
    followup = asyncio.run(service.chat_async("那它多少钱？", "buyer_chat_a"))
    other_chat = asyncio.run(service.chat_async("这个多少钱？", "buyer_chat_b"))

    assert switched["item_id"] == followup["item_id"] == "DEMO_ITEM_002"
    assert followup["answer"] == "这件已经出掉了。"
    assert other_chat["action"] == "handoff"
    assert other_chat.get("item_id") is None


def test_unknown_explicit_item_does_not_fall_back_to_remembered_item() -> None:
    service = _service()
    asyncio.run(
        service.chat_async("多少钱？", "buyer_chat_missing", item_id="DEMO_ITEM_001")
    )

    result = asyncio.run(
        service.chat_async("这个多少钱？", "buyer_chat_missing", item_id="XXX999")
    )

    assert result["action"] == "handoff"
    assert result["item_id"] == "XXX999"
    assert result["answer"] == "稍等我看看"
    assert "XXX999" not in str(result["answer"])


def test_unknown_text_item_id_does_not_fall_back_to_remembered_item() -> None:
    service = _service()
    asyncio.run(
        service.chat_async("多少钱？", "buyer_chat_text_missing", item_id="DEMO_ITEM_001")
    )

    result = asyncio.run(
        service.chat_async(
            "请查商品编号 XXX999 的价格",
            "buyer_chat_text_missing",
        )
    )

    assert result["action"] == "handoff"
    assert result["item_id"] == "XXX999"
    assert result["answer"] == "稍等我看看"
    assert "XXX999" not in str(result["answer"])
