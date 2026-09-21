"""Conversation-control API tests with an isolated channel store."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.api import conversations
from app.channels.xianyu.store import ChannelStore
from app.main import app


def test_resume_auto_only_changes_the_requested_human_conversation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = ChannelStore(tmp_path / "temporary-channel.sqlite3")
    store.set_session_mode("seller", "target-chat", "buyer-1", "HUMAN", "need seller")
    store.set_session_mode("seller", "other-chat", "buyer-2", "HUMAN", "keep seller")
    monkeypatch.setitem(
        app.dependency_overrides,
        conversations.get_channel_store,
        lambda: store,
    )

    response = TestClient(app).post(
        "/conversations/target-chat/resume-auto",
        json={"account_id": "seller"},
    )

    assert response.status_code == 200
    assert response.json()["mode"] == "AUTO"
    assert store.session_state("seller", "target-chat", "buyer-1")["mode"] == "AUTO"
    assert store.session_state("seller", "other-chat", "buyer-2")["mode"] == "HUMAN"
