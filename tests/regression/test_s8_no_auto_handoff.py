"""S8 acceptance checks: automatic failures never request human takeover."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from app.channels.xianyu.models import InboundMessage, SendReceipt
from app.channels.xianyu.stage3_worker import XianyuStage3Worker
from app.channels.xianyu.store import ChannelStore
from app.rag.answerability import AnswerReliability
from app.services.xianyu.responses import clarification, item_conflict, unavailable


class _Sender:
    async def send_text(self, *args: object) -> SendReceipt:
        return SendReceipt(request_id=str(args[-1]), local_submitted=True, platform_confirmed=True)


class _Chat:
    def __init__(self, response: Mapping[str, Any] | BaseException) -> None:
        self.response = response

    async def ask(self, message: object) -> Mapping[str, Any]:
        del message
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


class _Notifier:
    async def notify_handoff(self, **kwargs: object) -> None:
        del kwargs


def _message(message_id: str) -> InboundMessage:
    return InboundMessage(
        account_id="seller",
        platform_message_id=message_id,
        chat_id="chat-1",
        buyer_id="buyer-1",
        text="这个商品怎么样？",
        platform_item_id="listing-1",
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"action": "handoff", "reason": "knowledge_evidence_unavailable"},
        {"action": "clarify", "reason": "item_context_unavailable"},
        {"action": "handoff", "reason": "expert_execution_failed"},
        TimeoutError("model timeout"),
    ],
    ids=["knowledge", "missing_item", "expert_failure", "model_exception"],
)
def test_automatic_failure_keeps_channel_session_auto(
    tmp_path: Path,
    payload: Mapping[str, Any] | BaseException,
) -> None:
    store = ChannelStore(tmp_path / "channel.sqlite3")
    store.bind_item("seller", "listing-1", "ITEM-1")
    store.set_enabled("seller", True)
    worker = XianyuStage3Worker(
        store,
        account_id="seller",
        chat_client=_Chat(payload),  # type: ignore[arg-type]
        notifier=_Notifier(),  # type: ignore[arg-type]
    )

    result = asyncio.run(worker.process(_message("m-" + str(type(payload).__name__)), _Sender()))

    assert result["action"] in {"answer", "clarify", "error"}
    assert store.session_state("seller", "chat-1", "buyer-1")["mode"] == "AUTO"


def test_insufficient_item_responses_and_rag_reliability_do_not_request_human() -> None:
    for response in (clarification("缺商品"), item_conflict("商品冲突", "冲突")):
        assert response["action"] != "handoff"
        assert response["next_step"] != "human_handoff"

    reliability = AnswerReliability(threshold=0.5).evaluate([])
    assert reliability["next_step"] == "clarify"


def test_knowledge_unavailable_is_a_safe_reply_not_a_clarification() -> None:
    response = unavailable(
        "这个修过吗",
        "已确认：标价 100 元。\n暂无可确认的商品知识补充信息。",
    )

    assert response["action"] == "reply"
    assert response["can_answer"] is False
    assert "稍等我看看" not in response["answer"]


def test_unavailable_response_keeps_the_supplied_text() -> None:
    response = unavailable("售后怎么处理？", "模型原始回答")

    assert response["action"] == "reply"
    assert response["answer"] == "模型原始回答"
