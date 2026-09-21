from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from app.api import chat as chat_api
from app.main import app


client = TestClient(app)


def test_chat_post_returns_retrieval_response(monkeypatch) -> None:
    service_chat = AsyncMock(
        return_value={
            "query": "订单一般多久发货？",
            "chat_id": "chat_general_001",
            "results": [],
        }
    )
    monkeypatch.setattr(chat_api.chat_service, "chat_async", service_chat)

    response = client.post(
        "/chat",
        json={"query": "订单一般多久发货？", "chat_id": "chat_general_001"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "query": "订单一般多久发货？",
        "chat_id": "chat_general_001",
        "results": [],
    }


def test_chat_calls_chat_service(monkeypatch) -> None:
    service_chat = AsyncMock(
        return_value={
            "query": "订单状态是什么？",
            "chat_id": "chat_general_002",
            "results": [],
        }
    )
    monkeypatch.setattr(chat_api.chat_service, "chat_async", service_chat)

    response = client.post(
        "/chat",
        json={"query": "订单状态是什么？", "chat_id": "chat_general_002"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "query": "订单状态是什么？",
        "chat_id": "chat_general_002",
        "results": [],
    }
    service_chat.assert_awaited_once_with(
        "订单状态是什么？",
        "chat_general_002",
        item_id=None,
    )


def test_chat_requires_query() -> None:
    response = client.post("/chat", json={})

    assert response.status_code == 422


def test_chat_rejects_blank_query() -> None:
    response = client.post(
        "/chat",
        json={"query": " \t\n ", "chat_id": "chat_blank_query"},
    )

    assert response.status_code == 422


def test_chat_strips_query_before_dispatch(monkeypatch) -> None:
    service_chat = AsyncMock(return_value={"query": "shipping", "results": []})
    monkeypatch.setattr(chat_api.chat_service, "chat_async", service_chat)

    response = client.post(
        "/chat",
        json={"query": "  shipping  ", "chat_id": "chat_trim_query"},
    )

    assert response.status_code == 200
    service_chat.assert_awaited_once_with("shipping", "chat_trim_query", item_id=None)


def test_chat_requires_chat_id() -> None:
    response = client.post("/chat", json={"query": "订单一般多久发货？"})

    assert response.status_code == 422


def test_chat_does_not_accept_get() -> None:
    response = client.get("/chat")

    assert response.status_code == 405
