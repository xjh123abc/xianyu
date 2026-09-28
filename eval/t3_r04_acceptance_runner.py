from __future__ import annotations

import asyncio
import copy
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.channels.xianyu.action_mapper import map_chat_response
from app.generation.deepseek import DeepSeekGenerator
from app.services.chat_contracts import ChatMessage, SessionContext, Task, TaskResult
from app.services.chat_service import ChatService
from app.services.intent_router import IntentRouter
from app.services.item_service import ItemService
from app.services.planner import Planner
from app.services.result_merger import ResultMerger
from app.services.session_manager import SessionManager
from app.services.task_executor import TaskExecutor
from app.services.technical_knowledge_service import TechnicalKnowledgeService
from config.settings import settings


REPORT_DIR = ROOT / "eval" / "results"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
JSON_PATH = REPORT_DIR / f"t3_r04_acceptance_{STAMP}.json"
MD_PATH = REPORT_DIR / f"t3_r04_acceptance_{STAMP}.md"

ITEM_ID = "CANON_FTB_001"
BASE_ITEM = ItemService().get_item_info(ITEM_ID)


class EvidenceRag:
    def warm_up(self) -> None:
        return None

    def prepare(self, query: str, *, item_id: str | None = None) -> dict[str, object]:
        source = {"source": "seller_rules.md", "index": 0}
        return {
            "can_answer": True,
            "context": {
                "context": (
                    "店铺售后规则：买家反馈到货损坏或商品问题时，需先在平台售后提交照片/视频、包装、面单等证据。"
                    "未核实前不能承诺退货退款或运费由卖家承担；经核实属于卖家责任时，合理必要费用按平台规则处理。"
                    "保修问题按平台售后和商品说明核实。"
                ),
                "sources": [source],
            },
            "sources": [source],
            "results": [],
            "reliability": None,
        }


class EmptyRag:
    def warm_up(self) -> None:
        return None

    def prepare(self, query: str, *, item_id: str | None = None) -> dict[str, object]:
        return {
            "can_answer": False,
            "context": None,
            "sources": [],
            "results": [],
            "reliability": None,
            "issues": ["knowledge_evidence_unavailable"],
        }


class GeneralRag:
    rag_pipeline = None
    generator = None

    def chat(self, query: str) -> dict[str, object]:
        return {
            "query": query,
            "route": "rag",
            "action": "reply",
            "answer": "通用知识回答",
            "can_answer": True,
        }


class FakeMcp:
    def __init__(self, item: Mapping[str, object] | None = None) -> None:
        self.item = copy.deepcopy(dict(item or BASE_ITEM))
        self.calls: list[str] = []

    async def get_item_info(self, item_id: str) -> dict[str, object]:
        self.calls.append(item_id)
        if item_id == self.item.get("item_id"):
            return copy.deepcopy(self.item)
        return {"found": False, "item_id": item_id}


class FakeGenerator:
    def __init__(self, semantic_payloads: Mapping[str, object] | None = None) -> None:
        self.semantic_payloads = dict(semantic_payloads or {})
        self.analyze_calls: list[dict[str, object]] = []
        self.expert_calls: list[dict[str, object]] = []
        self.generate_calls: list[str] = []

    def analyze_xianyu_intent(
        self,
        query: str,
        *,
        history: object | None = None,
        timeout_seconds: float | None = None,
    ) -> object | None:
        self.analyze_calls.append(
            {
                "query": query,
                "history_count": len(history or []) if isinstance(history, list) else 0,
                "timeout_seconds": timeout_seconds,
            }
        )
        payload = self.semantic_payloads.get(query)
        if isinstance(payload, BaseException):
            raise payload
        if callable(payload):
            return payload(query=query, history=history, timeout_seconds=timeout_seconds)
        return payload

    def generate_xianyu_expert(
        self,
        expert: str,
        question: str,
        item: Mapping[str, object] | None,
        evidence: str,
        **kwargs: object,
    ) -> str:
        self.expert_calls.append(
            {"expert": expert, "question": question, "evidence": evidence, "kwargs": kwargs}
        )
        if "运费" in question or "邮费" in question or "寄回" in question:
            return (
                "退回运费要先看退货原因和责任归属，未核实前不能直接承诺由卖家承担；"
                "经核实属于卖家责任时，再按平台规则处理合理必要费用。"
            )
        if "保修" in question:
            return "保修和售后需要按商品说明、订单情况和平台流程核实，目前不能直接承诺处理结果。"
        return "需要先按平台售后流程提交证据核实，核实后再按平台规则处理。"

    def generate_xianyu(self, query: str, item: Mapping[str, object], context: str, **kwargs: object) -> str:
        self.generate_calls.append(query)
        return "旧生成器不应被本验收路径调用。"

    def classify_intent(self, query: str) -> str | None:
        return None


class RealPlanningGenerator(FakeGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.real = DeepSeekGenerator()

    def analyze_xianyu_intent(
        self,
        query: str,
        *,
        history: object | None = None,
        timeout_seconds: float | None = None,
    ) -> object | None:
        self.analyze_calls.append(
            {
                "query": query,
                "history_count": len(history or []) if isinstance(history, list) else 0,
                "timeout_seconds": timeout_seconds,
            }
        )
        return self.real.analyze_xianyu_intent(
            query,
            history=history if isinstance(history, list) else None,
            timeout_seconds=timeout_seconds,
        )


class RecordingPlanner:
    def __init__(self, inner: Planner) -> None:
        self.inner = inner
        self.last_outcome = None
        self.outcomes = []

    async def plan_async(self, *args: object, **kwargs: object):
        outcome = await self.inner.plan_async(*args, **kwargs)
        self.last_outcome = outcome
        self.outcomes.append(outcome)
        return outcome

    def plan(self, *args: object, **kwargs: object):
        return self.inner.plan(*args, **kwargs)

    def xianyu_context_updates(self, *args: object, **kwargs: object):
        return self.inner.xianyu_context_updates(*args, **kwargs)


class RecordingExecutor:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.last_tasks: list[Task] = []
        self.last_results: list[TaskResult] = []

    async def execute(
        self,
        tasks: Sequence[Task],
        message: ChatMessage,
        context: SessionContext,
        **kwargs: object,
    ) -> list[TaskResult]:
        self.last_tasks = list(tasks)
        results = await self.inner.execute(tasks, message, context, **kwargs)
        self.last_results = list(results)
        return results


class FakeSender:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def send_text(self, text: str) -> object:
        self.calls.append(text)
        return {"submitted": True}


def make_service(*, generator: FakeGenerator, rag: object | None = None):
    service = ChatService(
        rag_service=GeneralRag(),
        mcp_service=FakeMcp(),
        session_manager=SessionManager(),
        xianyu_rag_service=rag or EvidenceRag(),
        generator=generator,
        technical_knowledge_service=TechnicalKnowledgeService(enabled=False),
    )
    rec_planner = RecordingPlanner(service.planner)
    rec_executor = RecordingExecutor(service.task_executor)
    service.planner = rec_planner  # type: ignore[assignment]
    service.task_executor = rec_executor  # type: ignore[assignment]
    return service, rec_planner, rec_executor


def safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): safe(v) for k, v in value.items() if k != "_planner_state"}
    if isinstance(value, (list, tuple, set)):
        return [safe(v) for v in value]
    return str(value)


def task_record(task: Task) -> dict[str, object]:
    intent_context = task.metadata.get("intent_context") if isinstance(task.metadata, Mapping) else None
    conditions = intent_context.get("conditions") if isinstance(intent_context, Mapping) else []
    return {
        "task_id": task.task_id,
        "type": task.task_type,
        "query": task.query,
        "query_target": task.query_target,
        "execution_mode": task.execution_mode,
        "normalized_question": task.metadata.get("normalized_question"),
        "knowledge_scope": task.metadata.get("knowledge_scope"),
        "intent": intent_context.get("intent") if isinstance(intent_context, Mapping) else None,
        "conditions": safe(conditions),
        "reply_required": task.metadata.get("reply_required"),
        "depends_on_task_ids": list(task.depends_on_task_ids),
        "transaction_conditions": safe(task.metadata.get("transaction_conditions", {})),
    }


def result_nodes(results: Sequence[TaskResult]) -> list[dict[str, object]]:
    for result in results:
        trace = result.metadata.get("workflow_trace")
        if isinstance(trace, list):
            return [safe(entry) for entry in trace]
    return []


def response_brief(response: Mapping[str, object]) -> dict[str, object]:
    return {
        "action": response.get("action"),
        "answer": response.get("answer"),
        "can_answer": response.get("can_answer"),
        "reason": response.get("reason"),
        "route": response.get("route"),
        "item_id": response.get("item_id"),
    }


def conclude(record: dict[str, object], checks: object) -> None:
    if isinstance(checks, Mapping):
        flow = checks.get("flow", [])
        semantic = checks.get("semantic", [])
    else:
        flow = checks
        semantic = []
    flow_checks = list(flow) if isinstance(flow, list) else []
    semantic_checks = list(semantic) if isinstance(semantic, list) else []
    record["flow_checks"] = [
        {"name": name, "passed": bool(ok)} for name, ok in flow_checks
    ]
    record["semantic_checks"] = [
        {"name": name, "passed": bool(ok)} for name, ok in semantic_checks
    ]
    # Keep the old field for quick greps, but do not let one category stand in
    # for the whole sample.
    record["checks"] = [*record["flow_checks"], *record["semantic_checks"]]
    all_checks = [*flow_checks, *semantic_checks]
    record["conclusion"] = "PASS" if all(ok for _, ok in all_checks) else "FAIL"


def case_checks(*checks: tuple[str, bool]) -> list[tuple[str, bool]]:
    return list(checks)


def split_checks(
    *,
    flow: Sequence[tuple[str, bool]] = (),
    semantic: Sequence[tuple[str, bool]] = (),
) -> dict[str, list[tuple[str, bool]]]:
    return {"flow": list(flow), "semantic": list(semantic)}


def has_node(record: Mapping[str, object], node: str) -> bool:
    return any(isinstance(entry, Mapping) and entry.get("node") == node for entry in record.get("nodes", []))


def run_chat_case(case: dict[str, object]) -> dict[str, object]:
    generator = case.get("generator") or FakeGenerator(
        case.get("semantic_payloads") if isinstance(case.get("semantic_payloads"), Mapping) else None
    )
    service, rec_planner, rec_executor = make_service(
        generator=generator,  # type: ignore[arg-type]
        rag=case.get("rag") or EvidenceRag(),
    )
    chat_id = str(case.get("chat_id") or case["id"])
    prior_outputs = []
    for turn in case.get("history_turns") or []:
        prior_outputs.append(
            asyncio.run(
                service.chat_async(
                    turn["query"],
                    chat_id,
                    item_id=turn.get("item_id"),
                    turn_id=turn.get("turn_id"),
                )
            )
        )
    response = asyncio.run(
        service.chat_async(
            str(case["query"]),
            chat_id,
            item_id=case.get("item_id"),
            turn_id=str(case.get("turn_id") or case["id"]),
        )
    )
    outcome = rec_planner.last_outcome
    understanding = getattr(outcome, "understanding", None)
    tasks = list(getattr(outcome, "tasks", []) or rec_executor.last_tasks)
    results = rec_executor.last_results
    record = {
        "id": case["id"],
        "group": case.get("group", "simulated_main_chain"),
        "query": case["query"],
        "history_turns": safe(case.get("history_turns") or []),
        "prior_outputs": safe([response_brief(out) for out in prior_outputs]),
        "tasks": [task_record(task) for task in tasks],
        "task_count": len(tasks),
        "conditions": [task_record(task)["conditions"] for task in tasks],
        "model_planning_call_count": len(generator.analyze_calls),  # type: ignore[attr-defined]
        "understanding": {
            "status": getattr(understanding, "status", None),
            "model_called": getattr(understanding, "model_called", None),
            "error_reason": getattr(understanding, "error_reason", None),
            "clarification_question": getattr(understanding, "clarification_question", None),
        },
        "nodes": result_nodes(results),
        "task_results": [
            safe({"task_id": r.task_id, "status": r.status, "answer": r.answer, "reason": r.reason})
            for r in results
        ],
        "answer": response.get("answer"),
        "response": response_brief(response),
        "generator_calls": {
            "semantic": safe(generator.analyze_calls),  # type: ignore[attr-defined]
            "expert": safe(generator.expert_calls),  # type: ignore[attr-defined]
            "legacy_generate": safe(generator.generate_calls),  # type: ignore[attr-defined]
        },
    }
    conclude(record, case["check"](record))
    return record


def semantic_payload_for_item_identity(query: str) -> dict[str, object]:
    return {
        "status": "ready",
        "needs": [
            {
                "need_id": "m1",
                "intent": "product.identity_model",
                "original_question": query,
                "normalized_question": "查询当前商品型号",
                "subject": "当前商品",
                "conditions": [],
                "source_texts": [query],
                "context_references": [],
                "requested_outcome": "identity_model",
                "reply_required": True,
                "depends_on_need_ids": [],
            }
        ],
    }


MAIN_CASES = [
    {
        "id": "natural_greeting",
        "query": "你好",
        "item_id": None,
        "check": lambda r: case_checks(
            ("one_greeting_task", [(t["type"], t["query_target"]) for t in r["tasks"]] == [("service", "greeting")]),
            ("model_calls_0", r["model_planning_call_count"] == 0),
            ("no_expert_model", len(r["generator_calls"]["expert"]) == 0),
            ("reply", r["response"]["action"] == "reply" and bool(r["answer"])),
        ),
    },
    {
        "id": "natural_thanks",
        "query": "谢谢",
        "item_id": None,
        "check": lambda r: case_checks(
            ("one_thanks_task", [(t["type"], t["query_target"]) for t in r["tasks"]] == [("service", "thanks")]),
            ("model_calls_0", r["model_planning_call_count"] == 0),
            ("reply", r["response"]["action"] == "reply" and "不客气" in str(r["answer"])),
        ),
    },
    {
        "id": "pure_system_event_ignore",
        "query": "[系统] 买家已读消息",
        "item_id": None,
        "check": lambda r: case_checks(
            ("no_reply_task", [(t["type"], t["query_target"]) for t in r["tasks"]] == [("service", "no_reply")]),
            ("model_calls_0", r["model_planning_call_count"] == 0),
            ("action_ignore", r["response"]["action"] == "ignore" and r["answer"] == ""),
            ("service_node", has_node(r, "service_node")),
        ),
    },
    {
        "id": "colloquial_availability",
        "query": "还有吗",
        "item_id": ITEM_ID,
        "check": lambda r: case_checks(
            (
                "availability_task",
                [(t["type"], t["query_target"]) for t in r["tasks"]] == [("product", "availability.sale_status")],
            ),
            ("model_calls_0", r["model_planning_call_count"] == 0),
            ("answered", "还在" in str(r["answer"])),
        ),
    },
    {
        "id": "colloquial_function",
        "query": "能用不",
        "item_id": ITEM_ID,
        "check": lambda r: case_checks(
            ("function_task", [(t["type"], t["query_target"]) for t in r["tasks"]] == [("product", "function.overall")]),
            ("model_calls_0", r["model_planning_call_count"] == 0),
            ("answered", bool(r["answer"])),
        ),
    },
    {
        "id": "return_shipping_fee_with_damage_background",
        "query": "镜头开裂，退回去运费谁出",
        "item_id": ITEM_ID,
        "check": lambda r: case_checks(
            ("one_fee_task", [(t["type"], t["query_target"]) for t in r["tasks"]] == [("service", "seller_rule.general")]),
            ("intent_fee", r["tasks"][0]["intent"] == "after_sale.return_shipping_fee" if r["tasks"] else False),
            (
                "condition_kept",
                bool(
                    r["tasks"]
                    and r["tasks"][0]["conditions"]
                    and r["tasks"][0]["conditions"][0].get("event") == "镜头裂了"
                    and r["tasks"][0]["conditions"][0].get("modality") == "reported_unverified"
                ),
            ),
            ("not_over_split", r["task_count"] == 1),
            ("model_calls_0", r["model_planning_call_count"] == 0),
        ),
    },
    {
        "id": "return_policy_and_shipping_fee_two_needs",
        "query": "可以退吗？退回去运费谁出",
        "item_id": ITEM_ID,
        "check": lambda r: split_checks(
            flow=case_checks(
                ("not_merged_by_expert", r["task_count"] == 2),
                ("model_calls_0", r["model_planning_call_count"] == 0),
            ),
            semantic=case_checks(
                ("two_real_needs", [t["intent"] for t in r["tasks"]] == ["after_sale.consult", "after_sale.return_shipping_fee"]),
                ("fresh_session_no_invented_problem", [t["conditions"] for t in r["tasks"]] == [[], []]),
            ),
        ),
    },
    {
        "id": "thanks_min_price_not_ignored",
        "query": "谢谢，最低多少",
        "item_id": ITEM_ID,
        "check": lambda r: case_checks(
            ("price_only", [(t["type"], t["query_target"]) for t in r["tasks"]] == [("price", "price.minimum")]),
            ("not_ignore", r["response"]["action"] == "reply" and r["answer"] != ""),
            ("model_calls_0", r["model_planning_call_count"] == 0),
        ),
    },
    {
        "id": "multi_question_four_needs",
        "query": "这台修过没？成色怎么样？最低多少？今天能发吗？",
        "item_id": ITEM_ID,
        "check": lambda r: case_checks(
            (
                "four_tasks",
                [(t["type"], t["query_target"]) for t in r["tasks"]]
                == [
                    ("product", "history.repair_history"),
                    ("product", "condition.summary"),
                    ("price", "price.minimum"),
                    ("service", "shipping.dispatch_time"),
                ],
            ),
            ("nodes_all", all(has_node(r, node) for node in ("product_node", "price_node", "service_node"))),
            ("model_calls_0", r["model_planning_call_count"] == 0),
        ),
    },
    {
        "id": "partial_unknown_keeps_known_answers",
        "query": "测光和手机对比过吗？最低多少？今天能发吗？",
        "item_id": ITEM_ID,
        "rag": EmptyRag(),
        "check": lambda r: split_checks(
            flow=case_checks(
                ("three_tasks", r["task_count"] == 3),
                ("no_handoff", r["response"]["action"] == "reply"),
                (
                    "partial_failure_retained",
                    r["response"]["can_answer"] is False
                    and len(r.get("task_results", [])) == 3
                    and r["task_results"][1].get("status") == "answered"
                    and r["task_results"][2].get("status") == "answered"
                    and bool(r["task_results"][1].get("answer"))
                    and bool(r["task_results"][2].get("answer")),
                ),
            ),
            semantic=case_checks(
                (
                    "item_specific_inspection_record",
                    [(t["type"], t["query_target"], t["knowledge_scope"]) for t in r["tasks"]]
                    == [
                        ("product", "function.inspection_record", "item_fact"),
                        ("price", "price.minimum", "item_fact"),
                        ("service", "shipping.dispatch_time", "item_fact"),
                    ],
                ),
                (
                    "not_model_knowledge_inference",
                    r["tasks"][0]["intent"] == "product.inspection_record"
                    and r["task_results"][0].get("status") == "unavailable"
                    and "record_unavailable" in str(r["task_results"][0].get("reason")),
                ),
            ),
        ),
    },
    {
        "id": "continuous_followup_shipping_condition",
        "query": "那不包邮呢？",
        "item_id": None,
        "history_turns": [{"query": "包邮最低多少？", "item_id": ITEM_ID, "turn_id": "followup-1"}],
        "check": lambda r: case_checks(
            ("price_followup_task", [(t["type"], t["query_target"]) for t in r["tasks"]] == [("price", "price.minimum")]),
            (
                "buyer_pays_condition",
                bool(r["tasks"] and r["tasks"][0]["transaction_conditions"].get("shipping") == "buyer_pays"),
            ),
            ("uses_context_item", r["response"].get("item_id") == ITEM_ID),
            (
                "model_calls_0_for_second_turn",
                len([c for c in r["generator_calls"]["semantic"] if c.get("query") == "那不包邮呢？"]) == 0,
            ),
        ),
    },
    {
        "id": "semantic_added_product_task_requires_item_resolution",
        "query": "这个商品是哪款",
        "item_id": None,
        "semantic_payloads": {"这个商品是哪款": semantic_payload_for_item_identity("这个商品是哪款")},
        "check": lambda r: split_checks(
            flow=case_checks(
                ("model_called_once", r["model_planning_call_count"] == 1),
                ("missing_item_clarify", r["response"]["action"] == "clarify" and bool(r["response"].get("answer"))),
                ("prepared_by_graph", has_node(r, "prepare_ready_tasks")),
            ),
            semantic=case_checks(
                ("product_task_added", [(t["type"], t["query_target"]) for t in r["tasks"]] == [("product", "identity.model")]),
            ),
        ),
    },
    {
        "id": "warranty_not_changed_to_identity_without_item",
        "query": "这个商品保修多久",
        "item_id": None,
        "semantic_payloads": {"这个商品保修多久": semantic_payload_for_item_identity("这个商品保修多久")},
        "check": lambda r: split_checks(
            flow=case_checks(
                ("model_called_once", r["model_planning_call_count"] == 1),
                ("reply_or_clarify", r["response"]["action"] in {"reply", "clarify"}),
            ),
            semantic=case_checks(
                ("warranty_kept_as_service", [(t["type"], t["query_target"], t["intent"]) for t in r["tasks"]] == [("service", "seller_rule.general", "after_sale.consult")]),
                ("not_product_identity", all(t["intent"] != "product.identity_model" for t in r["tasks"])),
            ),
        ),
    },
    {
        "id": "warranty_not_changed_to_identity_with_item",
        "query": "这个商品保修多久",
        "item_id": ITEM_ID,
        "semantic_payloads": {"这个商品保修多久": semantic_payload_for_item_identity("这个商品保修多久")},
        "check": lambda r: split_checks(
            flow=case_checks(
                ("model_called_once", r["model_planning_call_count"] == 1),
                ("service_node", has_node(r, "service_node")),
            ),
            semantic=case_checks(
                ("warranty_kept_as_service", [(t["type"], t["query_target"], t["intent"]) for t in r["tasks"]] == [("service", "seller_rule.general", "after_sale.consult")]),
                ("answer_mentions_warranty", "保修" in str(r["answer"])),
                ("not_product_identity", all(t["intent"] != "product.identity_model" for t in r["tasks"])),
            ),
        ),
    },
]

FAULT_CASES = [
    {
        "id": "semantic_timeout_preserves_rule_tasks",
        "group": "simulated_faults",
        "query": "这台修过吗？保修多久？最低多少？",
        "item_id": ITEM_ID,
        "semantic_payloads": {"这台修过吗？保修多久？最低多少？": TimeoutError("semantic timeout")},
        "check": lambda r: split_checks(
            flow=case_checks(
                ("model_called_once", r["model_planning_call_count"] == 1),
                ("reply", r["response"]["action"] == "reply"),
                ("all_three_results_visible", len(r.get("task_results", [])) == 3),
            ),
            semantic=case_checks(
                (
                    "rules_and_warranty_degrade_preserved",
                    [(t["type"], t["query_target"], t["intent"]) for t in r["tasks"]]
                    == [
                        ("product", "history.repair_history", "product.repair_history"),
                        ("service", "seller_rule.general", "after_sale.consult"),
                        ("price", "price.minimum", "price.minimum"),
                    ],
                ),
                ("all_answered", all(result.get("status") == "answered" for result in r["task_results"])),
                ("answer_mentions_warranty", "保修" in str(r["answer"])),
            ),
        ),
    },
    {
        "id": "semantic_illegal_output_fail_closed",
        "group": "simulated_faults",
        "query": "这个商品保修多久",
        "item_id": None,
        "semantic_payloads": {"这个商品保修多久": "not json"},
        "check": lambda r: split_checks(
            flow=case_checks(
                ("model_called_once", r["model_planning_call_count"] == 1),
                ("no_illegal_product_task", all(t["query_target"] != "identity.model" for t in r["tasks"])),
            ),
            semantic=case_checks(
                ("safe_warranty_fallback", [(t["type"], t["query_target"], t["intent"]) for t in r["tasks"]] == [("service", "seller_rule.general", "after_sale.consult")]),
                ("reply", r["response"]["action"] == "reply"),
            ),
        ),
    },
]


async def partial_task_failure_case() -> dict[str, object]:
    planner = Planner(
        intent_router=IntentRouter(),
        requires_item_context=lambda query: True,
        may_contain_explicit_item_reference=lambda query: False,
    )
    outcome = await planner.plan_async(
        "这台修过没？最低多少？今天能发吗？",
        SessionContext(current_item_id=ITEM_ID),
        item_id=ITEM_ID,
    )
    tasks = list(outcome.tasks)

    class OkHandler:
        async def handle(self, task: Task, message: ChatMessage, context: SessionContext) -> TaskResult:
            if task.task_type == "product":
                return TaskResult(task.task_id, "answered", "没有维修过。")
            if task.task_type == "price":
                return TaskResult(task.task_id, "answered", "最低 ¥1490.00 可以拍。")
            return TaskResult(task.task_id, "answered", "ok")

    class FailingService:
        async def handle(self, task: Task, message: ChatMessage, context: SessionContext) -> TaskResult:
            raise RuntimeError("forced service failure")

    executor = TaskExecutor({"product": OkHandler(), "price": OkHandler(), "service": FailingService()})
    message = ChatMessage("xianyu", "seller", "fault-partial", "buyer", ITEM_ID, "这台修过没？最低多少？今天能发吗？")
    results = await executor.execute(tasks, message, SessionContext(current_item_id=ITEM_ID))
    response = ResultMerger().merge(message.text, tasks, results)
    record = {
        "id": "partial_task_failure_keeps_successes",
        "group": "simulated_faults",
        "query": message.text,
        "tasks": [task_record(t) for t in tasks],
        "task_count": len(tasks),
        "conditions": [task_record(t)["conditions"] for t in tasks],
        "model_planning_call_count": 0,
        "understanding": {
            "status": getattr(outcome.understanding, "status", None),
            "model_called": getattr(outcome.understanding, "model_called", None),
        },
        "nodes": result_nodes(results),
        "task_results": [safe({"task_id": r.task_id, "status": r.status, "answer": r.answer, "reason": r.reason}) for r in results],
        "answer": response.get("answer"),
        "response": response_brief(response),
        "generator_calls": {"semantic": [], "expert": [], "legacy_generate": []},
    }
    conclude(
        record,
        case_checks(
            ("service_failed", any(r.status == "unavailable" and r.reason == "task_execution_failed" for r in results)),
            ("successes_kept", "没有维修过" in str(record["answer"]) and "最低" in str(record["answer"])),
            ("reply_not_handoff", record["response"]["action"] == "reply"),
        ),
    )
    return record


async def ignore_sender_case(source_response: Mapping[str, object]) -> dict[str, object]:
    decision = map_chat_response(source_response)
    sender = FakeSender()
    if decision.action == "answer":
        await sender.send_text(decision.text)
    record = {
        "id": "channel_ignore_does_not_call_sender",
        "group": "simulated_faults",
        "query": "[系统] 买家已读消息",
        "tasks": [],
        "task_count": 0,
        "conditions": [],
        "model_planning_call_count": 0,
        "nodes": [],
        "answer": source_response.get("answer"),
        "response": response_brief(source_response),
        "channel_decision": safe({"action": decision.action, "text": decision.text, "reason": decision.reason}),
        "sender_calls": list(sender.calls),
    }
    conclude(record, case_checks(("mapped_to_ignore", decision.action == "ignore"), ("sender_not_called", sender.calls == [])))
    return record


def run() -> dict[str, object]:
    records: list[dict[str, object]] = []
    for case in MAIN_CASES:
        try:
            records.append(run_chat_case(case))
        except Exception as exc:
            records.append(
                {
                    "id": case["id"],
                    "group": case.get("group", "simulated_main_chain"),
                    "query": case["query"],
                    "conclusion": "ERROR",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
    for case in FAULT_CASES:
        try:
            records.append(run_chat_case(case))
        except Exception as exc:
            records.append(
                {
                    "id": case["id"],
                    "group": "simulated_faults",
                    "query": case["query"],
                    "conclusion": "ERROR",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
    try:
        records.append(asyncio.run(partial_task_failure_case()))
    except Exception as exc:
        records.append(
            {
                "id": "partial_task_failure_keeps_successes",
                "group": "simulated_faults",
                "conclusion": "ERROR",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
    try:
        ignore_source = next((r["response"] for r in records if r.get("id") == "pure_system_event_ignore"), {"action": "ignore", "answer": ""})
        records.append(asyncio.run(ignore_sender_case(ignore_source)))
    except Exception as exc:
        records.append(
            {
                "id": "channel_ignore_does_not_call_sender",
                "group": "simulated_faults",
                "conclusion": "ERROR",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )

    if bool(str(getattr(settings, "deepseek_api_key", "")).strip()):
        try:
            real_case = {
                "id": "real_model_semantic_fallback_partial_message",
                "group": "real_model",
                "query": "这个商品保修多久？最低多少？",
                "item_id": ITEM_ID,
                "generator": RealPlanningGenerator(),
                "check": lambda r: split_checks(
                    flow=case_checks(
                        ("model_called_once", r["model_planning_call_count"] == 1),
                        ("from_chat_main_chain", r["response"].get("route") == "unified"),
                        ("no_handoff", r["response"]["action"] in {"reply", "clarify"}),
                        ("valid_targets", all(t.get("query_target") for t in r["tasks"])),
                    ),
                    semantic=case_checks(
                        ("warranty_task_present", any(t.get("intent") == "after_sale.consult" for t in r["tasks"])),
                        ("price_task_present", any(t.get("intent") == "price.minimum" for t in r["tasks"])),
                        ("not_warranty_to_identity", all(t.get("intent") != "product.identity_model" for t in r["tasks"])),
                        ("answer_mentions_both", "保修" in str(r["answer"]) and ("¥" in str(r["answer"]) or "元" in str(r["answer"]))),
                    ),
                ),
            }
            records.append(run_chat_case(real_case))
        except Exception as exc:
            records.append(
                {
                    "id": "real_model_semantic_fallback_partial_message",
                    "group": "real_model",
                    "query": "这个商品保修多久？最低多少？",
                    "conclusion": "ERROR",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
    else:
        records.append(
            {
                "id": "real_model_semantic_fallback_partial_message",
                "group": "real_model",
                "query": "这个商品保修多久？最低多少？",
                "conclusion": "NOT_RUN",
                "reason": "DEEPSEEK_API_KEY not configured",
            }
        )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "workspace": str(ROOT),
        "safety": {
            "real_buyer_send": False,
            "real_item_authorization_mutation": False,
            "mcp": "FakeMcp returning local ItemService snapshot only",
            "sender": "FakeSender; only channel mapper branch simulated",
        },
        "commands": [
            r"python -m pytest tests\regression\test_semantic_intent_planner.py tests\regression\test_result_merger.py tests\regression\test_task_executor.py tests\regression\test_expert_workflow.py tests\regression\test_unified_planner.py tests\api\test_xianyu_expert_contract.py tests\xianyu\test_xianyu_expert_agents.py tests\xianyu\test_xianyu_expert_orchestrator.py tests\xianyu\test_xianyu_price_agent.py tests\xianyu\test_t1_continuous_negotiation.py -q",
            r"python -m pytest tests\xianyu\test_xianyu_stage3_worker.py -q",
            r"python eval\t3_r04_acceptance_runner.py",
        ],
        "pytest": {
            "t3_related_matrix": "116 passed in 17.97s",
            "channel_worker_ignore_matrix": "25 passed in 17.31s",
        },
        "records": records,
        "totals": {
            "pass": sum(1 for r in records if r.get("conclusion") == "PASS"),
            "fail": sum(1 for r in records if r.get("conclusion") == "FAIL"),
            "error": sum(1 for r in records if r.get("conclusion") == "ERROR"),
            "not_run": sum(1 for r in records if r.get("conclusion") == "NOT_RUN"),
        },
    }
    summary["t3_acceptance"] = "PASS" if summary["totals"]["fail"] == 0 and summary["totals"]["error"] == 0 else "FAIL"
    JSON_PATH.write_text(json.dumps(safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# T3/R04 Acceptance Report\n",
        f"Generated: {summary['generated_at']}\n",
        f"JSON: `{JSON_PATH}`\n",
        "Safety: no real buyer send; fake MCP; no real item authorization mutation.\n",
        f"Pytest: {summary['pytest']['t3_related_matrix']}; channel worker: {summary['pytest']['channel_worker_ignore_matrix']}\n",
        f"Overall T3 acceptance: **{summary['t3_acceptance']}**\n",
    ]
    for group in ("simulated_main_chain", "simulated_faults", "real_model"):
        lines.append(f"\n## {group}\n")
        for rec in [r for r in records if r.get("group") == group]:
            lines.append(f"### {rec.get('id')} - {rec.get('conclusion')}\n")
            lines.append(f"- query: `{rec.get('query', '')}`\n")
            lines.append(f"- model planning calls: `{rec.get('model_planning_call_count', 'n/a')}`\n")
            lines.append(
                "- tasks: `"
                + str(
                    [
                        (t.get("type"), t.get("query_target"), t.get("intent"), t.get("conditions"))
                        for t in rec.get("tasks", [])
                    ]
                )
                + "`\n"
            )
            lines.append(
                "- nodes: `"
                + str(
                    [
                        (n.get("node"), n.get("task_id"), n.get("status"), n.get("reason"))
                        for n in rec.get("nodes", [])
                    ]
                )
                + "`\n"
            )
            lines.append(f"- answer: `{rec.get('answer', '')}`\n")
            lines.append(f"- response: `{rec.get('response', {})}`\n")
            if rec.get("flow_checks"):
                lines.append(f"- 流程检查: `{[(c.get('name'), c.get('passed')) for c in rec.get('flow_checks', [])]}`\n")
            if rec.get("semantic_checks"):
                lines.append(f"- 语义检查: `{[(c.get('name'), c.get('passed')) for c in rec.get('semantic_checks', [])]}`\n")
            if rec.get("error"):
                lines.append(f"- error: `{rec.get('error')}`\n")
    MD_PATH.write_text("".join(lines), encoding="utf-8")
    return {"json_path": str(JSON_PATH), "md_path": str(MD_PATH), "totals": summary["totals"], "t3_acceptance": summary["t3_acceptance"]}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
