"""S2.5A semantic-understanding planner regression coverage."""

from __future__ import annotations

import asyncio

from app.services.chat_contracts import SessionContext, TaskResult
from app.services.intent_analyzer import IntentAnalyzer
from app.services.intent_contracts import UnderstandingResult, UserNeed
from app.services.intent_router import IntentRouter
from app.services.planner import Planner
from app.services.result_merger import ResultMerger


def _planner(semantic_planner=None) -> Planner:
    return Planner(
        intent_router=IntentRouter(),
        requires_item_context=lambda query: True,
        may_contain_explicit_item_reference=lambda query: "ITEM-" in query,
        intent_analyzer=IntentAnalyzer(semantic_planner=semantic_planner, timeout_seconds=0.5),
    )


def _plan(query: str, context: SessionContext | None = None):
    return asyncio.run(
        _planner().plan_async(
            query,
            context or SessionContext(current_item_id="TEST_CORE_ALIGNMENT_CAMERA"),
            item_id="TEST_CORE_ALIGNMENT_CAMERA",
        )
    )


def test_hypothetical_lens_damage_is_one_service_need_not_product_fact() -> None:
    outcome = _plan("如果镜头裂了可以退吗？")

    assert outcome.early_response is None
    assert [(task.task_type, task.query_target) for task in outcome.tasks] == [
        ("service", "seller_rule.general")
    ]
    assert outcome.tasks[0].metadata["intent_context"]["intent"] == "after_sale.consult"
    assert outcome.tasks[0].metadata["intent_context"]["subject"] == "镜头"
    assert outcome.tasks[0].metadata["intent_context"]["conditions"][0]["modality"] == "hypothetical"
    assert "假设镜头裂了" in outcome.tasks[0].metadata["normalized_question"]


def test_reported_received_lens_damage_is_not_rewritten_as_hypothetical() -> None:
    outcome = _plan("今天早上收到货后发现镜头开裂，可以退吗？")

    assert [(task.task_type, task.query_target) for task in outcome.tasks] == [
        ("service", "seller_rule.general")
    ]
    task = outcome.tasks[0]
    assert task.query == "今天早上收到货后发现镜头开裂，可以退吗"
    assert "买家反馈收到商品时镜头裂了" in task.metadata["normalized_question"]
    assert "假设" not in task.metadata["normalized_question"]
    condition = task.metadata["intent_context"]["conditions"][0]
    assert condition["modality"] == "reported_unverified"
    assert condition["event"] == "镜头裂了"


def test_product_lens_question_and_after_sale_question_stay_separate() -> None:
    outcome = _plan("带什么镜头？如果镜头裂了能退吗？")

    assert [(task.task_type, task.query_target) for task in outcome.tasks] == [
        ("product", "lens.details"),
        ("service", "seller_rule.general"),
    ]
    assert [
        task.metadata["intent_context"]["intent"] for task in outcome.tasks
    ] == ["product.lens_details", "after_sale.consult"]


def test_model_knowledge_question_stays_single_answer_scope() -> None:
    outcome = _plan("这个相机可以测光吗？")

    assert [(task.task_type, task.query_target) for task in outcome.tasks] == [
        ("product", "product.model_knowledge")
    ]
    assert outcome.tasks[0].query == "这个相机可以测光吗"
    assert outcome.tasks[0].metadata["intent_context"]["intent"] == "product.model_knowledge"


def test_greeting_and_thanks_use_service_templates_without_model_planning() -> None:
    calls: list[str] = []

    def semantic_planner(query: str, **kwargs: object) -> object:
        del kwargs
        calls.append(query)
        return None

    planner = _planner(semantic_planner)

    greeting = asyncio.run(
        planner.plan_async("你好", SessionContext(current_item_id="TEST_CORE_ALIGNMENT_CAMERA"), item_id="TEST_CORE_ALIGNMENT_CAMERA")
    )
    thanks = asyncio.run(
        planner.plan_async("谢谢", SessionContext(current_item_id="TEST_CORE_ALIGNMENT_CAMERA"), item_id="TEST_CORE_ALIGNMENT_CAMERA")
    )

    assert calls == []
    assert [(task.task_type, task.query_target, task.metadata["knowledge_scope"]) for task in greeting.tasks] == [
        ("service", "greeting", "greeting")
    ]
    assert [(task.task_type, task.query_target, task.metadata["knowledge_scope"]) for task in thanks.tasks] == [
        ("service", "thanks", "greeting")
    ]
    assert greeting.understanding is not None and greeting.understanding.model_called is False
    assert thanks.understanding is not None and thanks.understanding.model_called is False


def test_no_reply_turns_stay_structured_instead_of_general_rag() -> None:
    for query in ("好的", "[系统] 买家已读消息"):
        outcome = asyncio.run(
            _planner().plan_async(query, SessionContext(current_item_id=None), item_id=None)
        )

        assert outcome.early_response is None
        assert [(task.task_type, task.query_target) for task in outcome.tasks] == [
            ("service", "no_reply")
        ]
        assert outcome.tasks[0].metadata["knowledge_scope"] == "no_reply"
        assert outcome.tasks[0].metadata["reply_required"] is False
        assert outcome.tasks[0].metadata["intent_context"]["intent"] == "service.no_reply"


def test_mixed_thanks_and_price_keeps_business_need_only_without_model_planning() -> None:
    calls: list[str] = []

    def semantic_planner(query: str, **kwargs: object) -> object:
        del kwargs
        calls.append(query)
        return None

    outcome = asyncio.run(
        _planner(semantic_planner).plan_async(
            "谢谢，最低多少",
            SessionContext(current_item_id="TEST_CORE_ALIGNMENT_CAMERA"),
            item_id="TEST_CORE_ALIGNMENT_CAMERA",
        )
    )

    assert calls == []
    assert [(task.task_type, task.query_target) for task in outcome.tasks] == [
        ("price", "price.minimum")
    ]
    assert outcome.tasks[0].metadata["intent_context"]["intent"] == "price.minimum"


def test_colloquial_product_questions_route_to_item_fact_tasks() -> None:
    availability = _plan("还有吗")
    function = _plan("能用不")

    assert [(task.task_type, task.query_target) for task in availability.tasks] == [
        ("product", "availability.sale_status")
    ]
    assert [(task.task_type, task.query_target) for task in function.tasks] == [
        ("product", "function.overall")
    ]


def test_colloquial_listing_price_questions_route_to_price_facts() -> None:
    for query in ("咖啡机怎么卖的", "相机卖多少钱"):
        outcome = _plan(query)

        assert [(task.task_type, task.query_target) for task in outcome.tasks] == [
            ("price", "price.listed_price")
        ]


def test_sale_reason_question_routes_to_the_seller_supplied_fact() -> None:
    outcome = _plan("这台相机为什么卖？")

    assert [(task.task_type, task.query_target) for task in outcome.tasks] == [
        ("product", "product_info.sale_reason")
    ]


def test_partial_rule_hit_calls_semantic_planner_once_and_keeps_known_tasks() -> None:
    calls: list[str] = []

    def semantic_planner(query: str, **kwargs: object) -> object:
        del kwargs
        calls.append(query)
        return {
            "status": "ready",
            "needs": [
                {
                    "need_id": "m0",
                    "intent": "product.repair_history",
                    "original_question": "这台修过吗",
                    "normalized_question": "模型重复的维修记录问题",
                    "subject": "当前商品",
                    "conditions": [],
                    "source_texts": ["这台修过吗"],
                    "context_references": [],
                    "requested_outcome": "repair_history",
                    "reply_required": True,
                    "depends_on_need_ids": [],
                },
                {
                    "need_id": "m1",
                    "intent": "after_sale.consult",
                    "original_question": "保修多久",
                    "normalized_question": "咨询当前商品保修或售后规则",
                    "subject": "保修",
                    "conditions": [],
                    "source_texts": ["保修多久"],
                    "context_references": [],
                    "requested_outcome": "warranty_policy",
                    "reply_required": True,
                    "depends_on_need_ids": [],
                },
                {
                    "need_id": "m2",
                    "intent": "price.minimum",
                    "original_question": "最低多少",
                    "normalized_question": "模型重复的最低价问题",
                    "subject": "当前商品",
                    "conditions": [],
                    "source_texts": ["最低多少"],
                    "context_references": [],
                    "requested_outcome": "minimum_price",
                    "reply_required": True,
                    "depends_on_need_ids": [],
                },
            ],
        }

    outcome = asyncio.run(
        _planner(semantic_planner).plan_async(
            "这台修过吗？保修多久？最低多少？",
            SessionContext(current_item_id="TEST_CORE_ALIGNMENT_CAMERA"),
            item_id="TEST_CORE_ALIGNMENT_CAMERA",
        )
    )

    assert calls == ["这台修过吗？保修多久？最低多少？"]
    assert outcome.understanding is not None and outcome.understanding.model_called is True
    assert [(task.task_type, task.query_target) for task in outcome.tasks] == [
        ("product", "history.repair_history"),
        ("service", "seller_rule.general"),
        ("price", "price.minimum"),
    ]


def test_semantic_success_does_not_build_a_discarded_legacy_task_plan() -> None:
    calls: list[str] = []

    async def analyze(query, context, *, item_id=None):
        del context, item_id
        calls.append(query)
        return UnderstandingResult.ready(
            [
                UserNeed(
                    need_id="price-1",
                    intent="price.minimum",
                    original_question="最低多少",
                    normalized_question="查询当前商品最低可接受价格",
                    subject="当前商品",
                    requested_outcome="minimum_price",
                )
            ],
            model_called=True,
        )

    planner = _planner()
    planner._intent_analyzer.analyze = analyze  # type: ignore[method-assign]

    def fail_if_called(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("semantic success should not materialize the legacy plan")

    planner.plan = fail_if_called  # type: ignore[method-assign]
    outcome = asyncio.run(
        planner.plan_async(
            "这台修过吗？保修多久？最低多少？",
            SessionContext(current_item_id="TEST_CORE_ALIGNMENT_CAMERA"),
            item_id="TEST_CORE_ALIGNMENT_CAMERA",
        )
    )

    assert calls == ["这台修过吗？保修多久？最低多少？"]
    assert [(task.task_type, task.query_target) for task in outcome.tasks] == [
        ("price", "price.minimum")
    ]


def test_semantic_planner_timeout_preserves_rule_tasks() -> None:
    calls: list[str] = []

    def semantic_planner(query: str, **kwargs: object) -> object:
        del kwargs
        calls.append(query)
        raise TimeoutError("semantic timeout")

    outcome = asyncio.run(
        _planner(semantic_planner).plan_async(
            "这台修过吗？保修多久？最低多少？",
            SessionContext(current_item_id="TEST_CORE_ALIGNMENT_CAMERA"),
            item_id="TEST_CORE_ALIGNMENT_CAMERA",
        )
    )

    assert calls == ["这台修过吗？保修多久？最低多少？"]
    assert outcome.understanding is not None and outcome.understanding.model_called is True
    assert [(task.task_type, task.query_target) for task in outcome.tasks] == [
        ("product", "history.repair_history"),
        ("service", "seller_rule.general"),
        ("price", "price.minimum"),
    ]
    assert [task.metadata["intent_context"]["intent"] for task in outcome.tasks] == [
        "product.repair_history",
        "after_sale.consult",
        "price.minimum",
    ]
    assert outcome.tasks[1].query == "保修多久"
    assert outcome.tasks[1].metadata["normalized_question"] == "咨询当前商品保修或售后规则"


def test_vague_followup_without_history_returns_clarification() -> None:
    outcome = asyncio.run(
        _planner().plan_async("那怎么办？", SessionContext(), item_id=None)
    )

    assert outcome.tasks == []
    assert outcome.early_response is not None
    assert outcome.early_response["action"] == "clarify"
    assert outcome.early_response["reason"] == "intent_clarification_required"


def test_same_service_expert_keeps_distinct_after_sale_needs() -> None:
    outcome = _plan("镜头裂了能退吗？退回去运费谁出？")

    assert [task.task_type for task in outcome.tasks] == ["service", "service"]
    assert [
        task.metadata["intent_context"]["intent"] for task in outcome.tasks
    ] == ["after_sale.consult", "after_sale.return_shipping_fee"]


def test_generic_return_policy_and_shipping_fee_do_not_invent_damage_reason() -> None:
    outcome = _plan("可以退吗？退回去运费谁出？")

    assert [task.task_type for task in outcome.tasks] == ["service", "service"]
    assert [
        task.metadata["intent_context"]["intent"] for task in outcome.tasks
    ] == ["after_sale.consult", "after_sale.return_shipping_fee"]
    assert [task.metadata["intent_context"]["conditions"] for task in outcome.tasks] == [[], []]
    assert outcome.tasks[0].metadata["normalized_question"] == "咨询当前商品是否支持退货或售后处理"


def test_seller_measurement_record_routes_to_item_fact_not_model_knowledge() -> None:
    outcome = _plan("测光和手机对比过吗？最低多少？今天能发吗？")

    assert [(task.task_type, task.query_target, task.metadata["knowledge_scope"]) for task in outcome.tasks] == [
        ("product", "function.inspection_record", "item_fact"),
        ("price", "price.minimum", "item_fact"),
        ("service", "shipping.dispatch_time", "item_fact"),
    ]
    assert outcome.tasks[0].metadata["intent_context"]["intent"] == "product.inspection_record"
    assert outcome.tasks[0].metadata["normalized_question"] == "查询当前商品是否有测光与手机对比记录"


def test_model_cannot_change_warranty_question_to_product_identity() -> None:
    calls: list[str] = []

    def semantic_planner(query: str, **kwargs: object) -> object:
        del kwargs
        calls.append(query)
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

    with_item = asyncio.run(
        _planner(semantic_planner).plan_async(
            "这个商品保修多久",
            SessionContext(current_item_id="TEST_CORE_ALIGNMENT_CAMERA"),
            item_id="TEST_CORE_ALIGNMENT_CAMERA",
        )
    )
    missing_item = asyncio.run(
        _planner(semantic_planner).plan_async(
            "这个商品保修多久",
            SessionContext(current_item_id=None),
            item_id=None,
        )
    )

    assert calls == ["这个商品保修多久", "这个商品保修多久"]
    for outcome in (with_item, missing_item):
        assert [(task.task_type, task.query_target) for task in outcome.tasks] == [
            ("service", "seller_rule.general")
        ]
        assert outcome.tasks[0].metadata["intent_context"]["intent"] == "after_sale.consult"
        assert outcome.tasks[0].query == "这个商品保修多久"
        assert outcome.tasks[0].metadata["normalized_question"] == "咨询当前商品保修或售后规则"


def test_damage_background_attaches_to_return_shipping_fee_need_only() -> None:
    outcome = _plan("今天早上收到货后发现镜头开裂，退回去运费谁出？")

    assert [(task.task_type, task.query_target) for task in outcome.tasks] == [
        ("service", "seller_rule.general")
    ]
    task = outcome.tasks[0]
    assert task.metadata["intent_context"]["intent"] == "after_sale.return_shipping_fee"
    assert task.metadata["normalized_question"] == (
        "买家反馈收到商品时镜头裂了，在未核实前咨询退回商品时运费由谁承担"
    )
    condition = task.metadata["intent_context"]["conditions"][0]
    assert condition == {
        "type": "scenario",
        "event": "镜头裂了",
        "timing": "收到商品时",
        "modality": "reported_unverified",
    }


def test_reported_lens_damage_return_shipping_fee_without_received_time_keeps_condition() -> None:
    outcome = _plan("镜头裂了，寄回去的钱算谁的？")

    assert [(task.task_type, task.query_target) for task in outcome.tasks] == [
        ("service", "seller_rule.general")
    ]
    task = outcome.tasks[0]
    assert task.metadata["intent_context"]["intent"] == "after_sale.return_shipping_fee"
    assert task.metadata["normalized_question"] == (
        "买家反馈当前镜头裂了，在未核实前咨询退回商品时运费由谁承担"
    )
    condition = task.metadata["intent_context"]["conditions"][0]
    assert condition == {
        "type": "scenario",
        "event": "镜头裂了",
        "timing": "unspecified",
        "modality": "reported_unverified",
    }


def test_return_shipping_fee_followup_inherits_damage_only_from_history() -> None:
    fresh = _plan("那寄回去的钱呢？", SessionContext(current_item_id="TEST_CORE_ALIGNMENT_CAMERA"))
    with_history = _plan(
        "那寄回去的钱呢？",
        SessionContext(
            current_item_id="TEST_CORE_ALIGNMENT_CAMERA",
            history=[{"role": "user", "content": "今天收到后发现镜头开裂，可以退吗？"}],
        ),
    )

    assert fresh.tasks[0].metadata["intent_context"]["conditions"] == []
    inherited_condition = with_history.tasks[0].metadata["intent_context"]["conditions"][0]
    assert inherited_condition == {
        "type": "scenario",
        "event": "镜头裂了",
        "timing": "收到商品时",
        "modality": "reported_unverified",
    }


def test_reported_lens_damage_return_policy_and_shipping_fee_are_two_real_needs() -> None:
    outcome = _plan("今天收到后发现镜头开裂，可以退吗？退回去运费谁出？")

    assert [task.task_type for task in outcome.tasks] == ["service", "service"]
    assert [
        task.metadata["intent_context"]["intent"] for task in outcome.tasks
    ] == ["after_sale.consult", "after_sale.return_shipping_fee"]
    assert [task.query for task in outcome.tasks] == [
        "今天收到后发现镜头开裂，可以退吗",
        "退回去运费谁出",
    ]
    assert [
        task.metadata["normalized_question"] for task in outcome.tasks
    ] == [
        "买家反馈收到商品时镜头裂了，在未核实前确认当前支持的售后处理方式",
        "买家反馈收到商品时镜头裂了，在未核实前咨询退回商品时运费由谁承担",
    ]
    for task in outcome.tasks:
        condition = task.metadata["intent_context"]["conditions"][0]
        assert condition == {
            "type": "scenario",
            "event": "镜头裂了",
            "timing": "收到商品时",
            "modality": "reported_unverified",
        }


def test_no_punctuation_multi_intent_is_not_collapsed() -> None:
    outcome = _plan("这台修过没最低多少今天能发收到有问题咋办")

    assert [task.task_type for task in outcome.tasks] == [
        "product",
        "price",
        "service",
        "service",
    ]
    assert [task.query_target for task in outcome.tasks] == [
        "history.repair_history",
        "price.minimum",
        "shipping.dispatch_time",
        "seller_rule.general",
    ]


def test_conditional_trade_uses_internal_dependency_without_exposing_it() -> None:
    outcome = _plan("如果没修过，1900我就买。")

    assert [(task.task_type, task.query_target) for task in outcome.tasks] == [
        ("product", "history.repair_history"),
        ("price", "price.offer"),
    ]
    assert outcome.tasks[0].metadata["reply_required"] is False
    assert outcome.tasks[1].depends_on_task_ids == ("q1",)

    merged = ResultMerger().merge(
        "如果没修过，1900我就买。",
        outcome.tasks,
        [
            TaskResult("q1", "answered", "没有维修记录。"),
            TaskResult("q2", "answered", "1900 可以。"),
        ],
    )

    assert merged["answer"] == "1900 可以。"


def test_same_price_expert_keeps_multiple_distinct_price_needs() -> None:
    outcome = _plan("最低多少？1900行不行？还能再少吗？")

    assert [task.task_type for task in outcome.tasks] == ["price", "price", "price"]
    assert [task.query_target for task in outcome.tasks] == [
        "price.minimum",
        "price.offer",
        "price.additional_discount",
    ]
