"""Sampling controls for the local Xianyu /chat evaluation runner."""

from __future__ import annotations

import pytest

from eval import batch_chat_test


def test_run_batch_limit_runs_only_requested_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        batch_chat_test,
        "_post_chat",
        lambda *_args, **_kwargs: {
            "answer": "ok",
            "action": "reply",
            "can_answer": True,
            "route": "xianyu",
            "reason": None,
            "sources": [],
        },
    )

    results = batch_chat_test.run_batch(limit=10)

    assert [result["id"] for result in results] == [f"T{index:02}" for index in range(1, 11)]


@pytest.mark.parametrize("limit", [0, 61])
def test_run_batch_rejects_out_of_range_limit(limit: int) -> None:
    with pytest.raises(ValueError, match="limit must be"):
        batch_chat_test.run_batch(limit=limit)
