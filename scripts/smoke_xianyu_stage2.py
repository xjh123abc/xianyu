"""Real local Stage 2 smoke test without external model generation."""

from __future__ import annotations

import json
from collections.abc import Mapping

from fastapi.testclient import TestClient

from app.api import chat as chat_api
from app.main import app
from app.services.chat_service import ChatService


class _LocalGroundedGenerator:
    """Validate retrieved context and answer locally without calling DeepSeek."""

    def generate(self, query: str, context: str) -> str:
        if "配件" not in query:
            raise AssertionError("A deterministic Stage 2 case unexpectedly reached generation")
        required = ("原厂背带", "镜头前后盖", "通用电池")
        forbidden = ("机械键盘", "空格键", "咖啡机", "粉碗")
        if not all(term in context for term in required):
            raise AssertionError("Selected item context is incomplete")
        if any(term in context for term in forbidden):
            raise AssertionError("Another item's facts leaked into the selected item context")
        return "配件包括原厂背带、镜头前后盖和一个通用电池。"


CASES = (
    (
        {
            "query": "这个商品多少钱？",
            "scenario": "xianyu",
            "item_id": "DEMO_ITEM_001",
        },
        {"action": "reply", "item_id": "DEMO_ITEM_001", "answer_contains": "1280.00"},
    ),
    (
        {
            "query": "这个商品还有吗？",
            "scenario": "xianyu",
            "item_id": "DEMO_ITEM_002",
        },
        {"action": "reply", "item_id": "DEMO_ITEM_002", "answer_contains": "已售出"},
    ),
    (
        {
            "query": "这个商品状态如何？",
            "scenario": "xianyu",
            "item_id": "DEMO_ITEM_003",
        },
        {"action": "handoff", "item_id": "DEMO_ITEM_003", "answer_contains": "未知"},
    ),
    (
        {"query": "这个东西有什么配件？", "scenario": "xianyu"},
        {"action": "clarify", "answer_contains": "商品编号"},
    ),
    (
        {
            "query": "这个商品多少钱，带哪些配件？",
            "scenario": "xianyu",
            "item_id": "DEMO_ITEM_001",
        },
        {
            "action": "reply",
            "item_id": "DEMO_ITEM_001",
            "answer_contains": "1280.00",
            "source_contains": "items/DEMO_ITEM_001.md",
        },
    ),
)


def _assert_response(payload: Mapping[str, object], expected: Mapping[str, str]) -> None:
    for field in ("action", "item_id"):
        if field in expected and payload.get(field) != expected[field]:
            raise AssertionError(
                f"Unexpected {field}: expected {expected[field]!r}, "
                f"got {payload.get(field)!r}; response={dict(payload)!r}"
            )
    expected_text = expected["answer_contains"]
    if expected_text not in str(payload.get("answer", "")):
        raise AssertionError(
            f"Answer did not contain {expected_text!r}: {payload.get('answer')!r}"
        )
    expected_source = expected.get("source_contains")
    if expected_source is not None:
        sources = payload.get("sources", [])
        if not isinstance(sources, list) or not any(
            isinstance(source, Mapping) and source.get("source") == expected_source
            for source in sources
        ):
            raise AssertionError(f"Response did not include source {expected_source!r}")


def main() -> None:
    """Exercise FastAPI and the real local MCP subprocess with demo data."""

    original_service = chat_api.chat_service
    chat_api.chat_service = ChatService(generator=_LocalGroundedGenerator())
    try:
        with TestClient(app) as client:
            summaries = []
            for request, expected in CASES:
                response = client.post("/chat", json=request)
                response.raise_for_status()
                payload = response.json()
                _assert_response(payload, expected)
                summaries.append(
                    {
                        "item_id": payload.get("item_id"),
                        "action": payload.get("action"),
                        "can_answer": payload.get("can_answer"),
                    }
                )
        print(json.dumps({"status": "passed", "cases": summaries}, ensure_ascii=False))
    finally:
        chat_api.chat_service = original_service


if __name__ == "__main__":
    main()
