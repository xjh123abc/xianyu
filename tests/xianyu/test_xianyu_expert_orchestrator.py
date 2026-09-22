"""S5 unified Xianyu expert exit and main-chain contract tests."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import time
from unittest.mock import AsyncMock, Mock

from app.services.chat_service import ChatService
from app.services.item_service import ItemService
from app.services.session_manager import SessionManager
from app.channels.xianyu.action_mapper import map_chat_response


class NoRag:
    def warm_up(self) -> None:
        return None

    def prepare(self, *args: object, **kwargs: object) -> dict[str, object]:
        raise AssertionError("this scenario should use item facts only")


class EmptyRag:
    def warm_up(self) -> None:
        return None

    def prepare(self, *args: object, **kwargs: object) -> dict[str, object]:
        return {
            "can_answer": False,
            "context": None,
            "sources": [],
            "results": [],
            "reliability": None,
        }


class EvidenceRag:
    def warm_up(self) -> None:
        return None

    def prepare(self, *args: object, **kwargs: object) -> dict[str, object]:
        source = {"source": "seller_rules.md", "index": 0}
        return {
            "can_answer": True,
            "context": {"context": "已确认的本店售后规则", "sources": [source]},
            "sources": [source],
            "results": [],
            "reliability": None,
        }


class LowReliabilityEvidenceRag:
    """Retrieved text must reach the expert even when the old guard rejects it."""

    def warm_up(self) -> None:
        return None

    def prepare(self, *args: object, **kwargs: object) -> dict[str, object]:
        return {
            "can_answer": False,
            "context": {"context": "低分 rerank 仍检索到的售后规则", "sources": []},
            "sources": [],
            "results": [{"content": "低分 rerank 仍检索到的售后规则"}],
            "reliability": {"can_answer": False, "reason": "below_threshold"},
        }


def _service(
    item: dict[str, object],
    generator: Mock | None = None,
    rag_service: object | None = None,
) -> ChatService:
    mcp = Mock()
    mcp.get_item_info = AsyncMock(return_value=item)
    return ChatService(
        mcp_service=mcp,
        xianyu_rag_service=rag_service or NoRag(),
        generator=generator or Mock(),
        session_manager=SessionManager(),
    )


def _canon_item() -> dict[str, object]:
    return ItemService().get_item_info("CANON_FTB_001")


def test_compound_turn_uses_one_unified_exit_and_keeps_task_order() -> None:
    generator = Mock()
    service = _service(_canon_item(), generator)

    result = asyncio.run(
        service.chat_async(
            "还在吗？有没有维修过？不包邮最低多少？",
            "s5_compound",
            item_id="CANON_FTB_001",
        )
    )
    assert result["action"] == "reply", result
    assert result["can_answer"] is True
    assert "还在的" in str(result["answer"])
    assert "没有维修过" in str(result["answer"])
    assert "1470.00" in str(result["answer"])
    generator.generate_xianyu.assert_not_called()
    generator.generate_xianyu_expert.assert_not_called()


def test_price_and_repair_dependency_handoff_does_not_make_a_conditional_offer() -> None:
    item = deepcopy(ItemService().get_item_info("DEMO_ITEM_001"))
    service = _service(item)

    result = asyncio.run(
        service.chat_async(
            "如果没修过，1470不包邮我就买",
            "s5_dependency",
            item_id="DEMO_ITEM_001",
        )
    )

    assert result["action"] != "handoff", result
    assert result["answer"] == ""
    assert result["reason"]
    assert "1470" not in str(result["answer"])


def test_follow_up_price_context_is_persisted_and_used_by_main_chain() -> None:
    service = _service(_canon_item())

    first = asyncio.run(
        service.chat_async("包邮最低多少？", "s5_followup", item_id="CANON_FTB_001")
    )
    second = asyncio.run(service.chat_async("那不包邮呢？", "s5_followup"))

    assert first["action"] == second["action"] == "reply"
    assert "1490.00" in str(first["answer"])
    assert "1470.00" in str(second["answer"])
    assert service.session_manager.get_xianyu_context("s5_followup")[
        "recent_price_topic"
    ] == "minimum"


def test_unified_unavailable_reason_survives_http_serialization(monkeypatch) -> None:
    from fastapi.testclient import TestClient
    from app.api import chat as chat_api
    from app.main import app

    fake_service = Mock()
    fake_service.chat_async = AsyncMock(
        return_value={
            "query": "测光对比过吗？",
            "route": "xianyu",
            "action": "reply",
            "answer": "该问题目前暂无足够信息确认。",
            "can_answer": False,
            "next_step": None,
            "reason": "缺少测光对比记录",
        }
    )
    monkeypatch.setattr(chat_api, "chat_service", fake_service)

    response = TestClient(app).post(
        "/chat",
        json={"query": "测光对比过吗？", "chat_id": "s5_http"},
    )

    assert response.status_code == 200
    assert response.json()["reason"] == "缺少测光对比记录"


def test_missing_model_result_is_fail_closed() -> None:
    generator = Mock()
    generator.plan_xianyu_questions.return_value = {
        "tasks": [
            {
                "task_id": "measure",
                "expert": "product",
                "question_fragment": "测光对比过吗",
                "normalized_question": "测光对比记录",
                "knowledge_scope": "model_knowledge",
                "transaction_conditions": {},
                "depends_on_task_ids": [],
            }
        ]
    }
    service = _service(_canon_item(), generator, EmptyRag())

    result = asyncio.run(
        service.chat_async(
            "测光对比过吗？顺便还在吗？",
            "s5_missing_model_result",
            item_id="CANON_FTB_001",
        )
    )

    assert result["action"] != "handoff"
    assert "暂无足够信息确认" not in result["answer"]
    assert result["can_answer"] is False


def test_legacy_missing_action_becomes_auto_clarification() -> None:
    mapped = map_chat_response(
        {"answer": "", "can_answer": False, "reason": "expert_result_empty"}
    )

    assert mapped.action == "error"
    assert mapped.reason == "expert_result_empty"


def test_expert_budget_discards_late_batch_result() -> None:
    from app.services.xianyu.expert_orchestrator import XianyuExpertOrchestrator
    from app.services.xianyu.experts.contracts import ExpertResult

    class SlowProduct:
        async def run(self, tasks, context):
            await asyncio.sleep(0.3)
            return [ExpertResult.answered(task, "迟到的答案") for task in tasks]

    service = _service(_canon_item())
    orchestrator = XianyuExpertOrchestrator(
        fact_responder=service.item_fact_responder,
        knowledge_responder=service.xianyu_knowledge_responder,
        intent_router=service.intent_router,
        generator=Mock(),
        product_agent=SlowProduct(),
        budget_seconds=0.15,
    )

    result = asyncio.run(
        orchestrator.handle("还在吗？", item=_canon_item())
    )

    assert result["action"] != "handoff"
    assert result["answer"] == ""
    assert result["reason"] == "expert_processing_timeout"


def test_model_planner_uses_a_bounded_subbudget() -> None:
    from app.services.xianyu.expert_orchestrator import XianyuExpertOrchestrator

    generator = Mock()
    generator.plan_xianyu_questions.return_value = {"tasks": []}
    service = _service(_canon_item(), generator)
    orchestrator = XianyuExpertOrchestrator(
        fact_responder=service.item_fact_responder,
        knowledge_responder=service.xianyu_knowledge_responder,
        intent_router=service.intent_router,
        generator=generator,
    )

    planner = orchestrator._model_planner(time.monotonic() + 60)
    assert planner is not None
    planner("还在吗？有没有维修过？")

    timeout = generator.plan_xianyu_questions.call_args.kwargs["timeout_seconds"]
    assert 0 < timeout <= orchestrator._PLANNER_BUDGET_SECONDS


def test_current_item_context_is_reused_for_an_unclassified_follow_up() -> None:
    service = _service(_canon_item(), rag_service=EmptyRag())

    asyncio.run(
        service.chat_async("还在吗？", "s5_current_item", item_id="CANON_FTB_001")
    )
    result = asyncio.run(service.chat_async("测光对比过吗？", "s5_current_item"))

    assert result["action"] != "handoff"
    assert result["item_id"] == "CANON_FTB_001"
    assert "knowledge_evidence_unavailable" in str(result["reason"])


def test_common_seller_rule_also_uses_the_unified_orchestrator() -> None:
    generator = Mock()
    generator.generate_xianyu_expert.return_value = "按已确认的本店规则处理。"
    service = _service(_canon_item(), generator, EvidenceRag())

    result = asyncio.run(
        service.chat_async("你们店一般怎么处理售后？", "s5_common_rule")
    )

    assert result["action"] == "reply"
    assert result["answer"] == "按已确认的本店规则处理。"
    generator.generate_xianyu.assert_not_called()
    generator.generate_xianyu_expert.assert_called_once()


def test_low_reliability_evidence_reaches_model_and_is_not_rewritten() -> None:
    generator = Mock()
    model_text = "模型原始回答：按低分检索到的规则处理。"
    generator.generate_xianyu_expert.return_value = model_text
    service = _service(_canon_item(), generator, LowReliabilityEvidenceRag())

    result = asyncio.run(
        service.chat_async("你们店售后怎么处理？", "safety_chain_removed")
    )

    evidence = generator.generate_xianyu_expert.call_args.args[3]
    assert "低分 rerank 仍检索到的售后规则" in evidence
    assert result["evidence"] == evidence
    assert result["raw_answer"] == model_text
    assert result["answer"] == model_text
    assert result["can_answer"] is True
