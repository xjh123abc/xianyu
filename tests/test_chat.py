from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.api import chat as chat_api
from app.main import app


client = TestClient(app)


def test_chat_post_returns_string_response() -> None:
    response = client.post("/chat", json={"query": "我的订单什么时候发货？"})

    assert response.status_code == 200
    assert response.json() == {"answer": "已收到你的问题：我的订单什么时候发货？"}


def test_chat_calls_chat_service(monkeypatch) -> None:
    service_chat = Mock(return_value="service answer")
    monkeypatch.setattr(chat_api.chat_service, "chat", service_chat)

    response = client.post("/chat", json={"query": "订单状态是什么？"})

    assert response.status_code == 200
    assert response.json() == {"answer": "service answer"}
    service_chat.assert_called_once_with("订单状态是什么？")


def test_chat_requires_query() -> None:
    response = client.post("/chat", json={})

    assert response.status_code == 422


def test_chat_does_not_accept_get() -> None:
    response = client.get("/chat")

    assert response.status_code == 405
