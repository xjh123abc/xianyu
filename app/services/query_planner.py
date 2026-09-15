"""Evidence planning and validated task planning for Xianyu experts."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Literal, TypedDict

from app.services.intent_router import IntentRouter, is_simple_single_question
from app.services.xianyu.experts.contracts import ExpertTask


logger = logging.getLogger(__name__)

ItemField = Literal["listed_price_cents", "sale_status", "lens", "included_items", "condition", "history", "identity"]
KnowledgeScope = Literal["common", "item"]
ExpertPlanner = Callable[..., str | Mapping[str, object]]


class KnowledgeQuestion(TypedDict):
    question: str
    scope: KnowledgeScope


class QuestionPlan(TypedDict):
    item_fields: list[ItemField]
    knowledge_questions: list[KnowledgeQuestion]


_PRICE_TERMS = ("价格", "多少钱", "标价", "售价", "多少元", "price", "cost")
_BARGAIN_TERMS = ("最低", "便宜", "少一点", "少点", "优惠", "小刀", "刀吗", "还价", "报价")
_BUYER_PAYS_TERMS = ("不包邮", "不用包邮", "出邮费", "出运费", "自付运费", "承担运费")
_SHIPPING_PRICE_TERMS = ("运费", "邮费", "快递费", "shipping fee")
_NON_ITEM_STATUS_TERMS = ("订单状态", "物流状态", "快递状态", "发货状态")
_SELLER_SCOPE_TERMS = ("你们店", "店里", "本店", "卖家", "这件商品", "这个商品")
_STATUS_TERMS = ("在吗", "能买吗", "还能买", "可买吗", "还有吗", "在售", "还没卖", "没卖", "还没出", "卖出", "售出", "已售", "状态", "available", "sold")

COMMON_KNOWLEDGE_RETRIEVAL_HINT = "卖家通用规则 售后与边界 二手商品 成色 瑕疵 配件 双方确认 争议 附加条件 单独确认 不自行承诺"
_COMMON_KNOWLEDGE_TERMS = ("售后", "质量问题", "退货", "退款", "发货", "运费", "邮费", "包邮", "规则", "政策", "shipping")
_ITEM_KNOWLEDGE_TERMS = ("配件", "包含", "附带", "成色", "瑕疵", "磕碰", "功能", "检测", "维修", "拆修", "改装", "使用", "续航", "说明", "accessory", "condition", "repair")
_ITEM_REFERENCE_TERMS = ("这个商品", "这件商品", "这个东西", "这件", "这台", "它")
_LENS_TERMS = ("镜头", "焦段", "焦距", "lens")
_INCLUDED_ITEMS_TERMS = ("配件", "包含", "附带", "带什么", "一起出", "赠送", "accessory")
_CONDITION_TERMS = ("成色", "外观", "瑕疵", "快门", "过片", "功能", "检测", "condition")
_HISTORY_TERMS = ("维修", "修过", "拆修", "改装", "摔", "磕碰", "维修记录", "repair")
_IDENTITY_TERMS = ("型号", "品牌", "品类", "类别", "model", "brand")
_DISPATCH_TERMS = ("今天能发", "什么时候发", "什么时候能发", "多久发", "发货", "从哪里发")
_CARRIER_TERMS = ("快递", "顺丰", "中通", "圆通", "韵达", "京东")
_AFTER_SALE_TERMS = ("闲鱼交易", "可以退", "退货", "退款", "收到发现", "描述一样", "保证", "确认收货", "验货")
_FOLLOW_UP_TERMS = ("那不包邮", "不包邮呢", "再少", "再便宜", "再优惠", "再刀")
_OFFER_PATTERN = re.compile(r"(?<!\d)[¥￥]?\s*(\d{1,7}(?:\.\d{1,2})?)\s*(?:元|块|rmb)?")
_VALID_EXPERTS = {"product", "price", "service"}
_VALID_SCOPES = {"item_fact", "model_knowledge", "seller_rule", "greeting"}
_EXPERT_BOUNDARY_TERMS = (
    "还在", "还没出", "还没卖", "卖掉", "有货", "在售",
    "维修", "修过", "拆修", "改装", "摔", "磕碰",
    "快门", "功能", "划痕", "毛病", "瑕疵", "成色", "外观",
    "配件", "带什么", "说明书", "包装", "镜头", "型号", "新手",
    "不包邮", "不用包邮", "包邮", "最低", "便宜", "优惠", "小刀",
    "发货", "快递", "顺丰", "售后", "退货", "退款",
)


@dataclass(frozen=True, slots=True)
class _Draft:
    task: ExpertTask
    position: int
    order: int


def _asks_item_price(query: str) -> bool:
    lowered_query = str(query or "").casefold()
    explicitly_item_price = any(term in lowered_query for term in ("商品价格", "商品多少钱", "标价", "售价"))
    if not explicitly_item_price and any(term in lowered_query for term in _SHIPPING_PRICE_TERMS):
        return False
    return any(term in lowered_query for term in _PRICE_TERMS)


def _asks_item_status(query: str) -> bool:
    lowered_query = str(query or "").casefold()
    return not any(term in lowered_query for term in _NON_ITEM_STATUS_TERMS) and any(term in lowered_query for term in _STATUS_TERMS)


def _structured_item_fields(query: str) -> list[ItemField]:
    lowered_query = str(query or "").casefold()
    fields: list[ItemField] = []
    for field, terms in (("lens", _LENS_TERMS), ("included_items", _INCLUDED_ITEMS_TERMS), ("condition", _CONDITION_TERMS), ("history", _HISTORY_TERMS), ("identity", _IDENTITY_TERMS)):
        if any(term in lowered_query for term in terms):
            fields.append(field)
    return fields


def build_question_plan(query: str) -> QuestionPlan:
    """Return the legacy fact/knowledge evidence plan without expert tasks."""

    lowered_query = str(query or "").casefold()
    item_fields: list[ItemField] = []
    if _asks_item_price(lowered_query):
        item_fields.append("listed_price_cents")
    if _asks_item_status(lowered_query):
        item_fields.append("sale_status")
    item_fields.extend(_structured_item_fields(lowered_query))
    clauses = [part.strip() for part in re.split(r"[，,。.!！?？、；;]+", str(query or "")) if part.strip()]
    knowledge_questions: list[KnowledgeQuestion] = []
    for clause in clauses:
        lowered_clause = clause.casefold()
        scopes: list[KnowledgeScope] = []
        if any(term in lowered_clause for term in _COMMON_KNOWLEDGE_TERMS):
            scopes.append("common")
        structured_fields = _structured_item_fields(lowered_clause)
        if any(term in lowered_clause for term in _ITEM_KNOWLEDGE_TERMS) and not structured_fields:
            scopes.append("item")
        if not scopes and not structured_fields and not _asks_item_price(lowered_clause) and not _asks_item_status(lowered_clause) and any(term in lowered_clause for term in _ITEM_REFERENCE_TERMS):
            scopes.append("item")
        for scope in scopes:
            need = {"question": clause, "scope": scope}
            if need not in knowledge_questions:
                knowledge_questions.append(need)
    return {"item_fields": item_fields, "knowledge_questions": knowledge_questions}


def build_expert_plan(
    query: str,
    *,
    intent_router: IntentRouter | None = None,
    planner: ExpertPlanner | None = None,
    history: Sequence[Mapping[str, object]] | None = None,
    xianyu_context: Mapping[str, object] | None = None,
) -> list[ExpertTask]:
    """Return every validated expert task for one buyer turn.

    Rules always retain known needs. A complex or follow-up turn makes at most
    one planner call; untrusted planner data can only add validated tasks.
    """

    normalized = str(query or "").strip()
    if not normalized:
        return []
    router = intent_router or IntentRouter()
    rule_drafts = _rule_drafts(normalized, xianyu_context)
    model_drafts: list[_Draft] = []
    if planner is not None and _requires_model_planning(normalized, xianyu_context):
        model_drafts = _validated_model_drafts(_call_planner(planner, normalized, history), normalized)
    return _finalize_drafts(_merge_drafts(model_drafts, rule_drafts, router))


def _rule_drafts(query: str, context: Mapping[str, object] | None) -> list[_Draft]:
    lowered = query.casefold()
    drafts: list[_Draft] = []

    def add(expert: str, fragment: str, normalized_question: str, scope: str, conditions: Mapping[str, object] | None = None) -> None:
        task = ExpertTask(f"rule-{len(drafts) + 1}", expert, fragment, normalized_question, scope, dict(conditions or {}))  # type: ignore[arg-type]
        drafts.append(_Draft(task, _position(query, fragment), len(drafts)))

    if _asks_item_status(query):
        add("product", _fragment(query, ("还在", "还没出", "还没卖", "还能拍", "卖掉", "有货", "在售")), "是否在售", "item_fact")
    if any(term in lowered for term in _HISTORY_TERMS):
        add("product", _fragment(query, _HISTORY_TERMS), _history_question(lowered), "item_fact")
    if any(term in lowered for term in ("快门", "功能", "正常使用", "直接用", "坏的", "坏了")):
        add("product", _fragment(query, ("快门", "功能", "正常使用", "直接用", "坏的", "坏了")), "功能是否正常", "item_fact")
    if any(term in lowered for term in ("划痕", "磕碰", "毛病", "隐藏问题", "瑕疵", "问题吗")):
        add("product", _fragment(query, ("划痕", "磕碰", "毛病", "隐藏问题", "瑕疵", "问题吗")), "商品瑕疵情况", "item_fact")
    if any(term in lowered for term in ("成色", "外观", "新不新", "使用痕迹")):
        add("product", _fragment(query, ("成色", "外观", "新不新", "使用痕迹")), "商品成色", "item_fact")
    if any(term in lowered for term in ("配件", "带什么", "包含", "一起给", "说明书", "包装", "镜头")):
        add("product", _fragment(query, ("配件", "带什么", "包含", "一起给", "说明书", "包装", "镜头")), "商品配件", "item_fact")
    if any(term in lowered for term in ("型号", "哪一年", "哪年生产", "新手", "怎么用", "为什么卖", "为什么要卖")):
        scope = "model_knowledge" if "怎么用" in lowered and "型号" in lowered else "item_fact"
        add("product", _fragment(query, ("型号", "哪一年", "哪年生产", "新手", "怎么用", "为什么卖", "为什么要卖")), "商品型号或使用信息", scope)

    price_conditions = _price_conditions(query, context)
    if _is_price_question(query, context):
        kind = str(price_conditions.get("request_kind", "listed_price"))
        question = {"minimum": "最低价", "offer": "买家报价", "additional_discount": "继续优惠", "listed_price": "商品标价"}[kind]
        add("price", _price_fragment(query), question, "item_fact", price_conditions)

    if any(term in lowered for term in _DISPATCH_TERMS):
        add("service", _fragment(query, _DISPATCH_TERMS), "发货时限或地点", "item_fact")
    if any(term in lowered for term in _CARRIER_TERMS) and not _is_price_question(query, context):
        add("service", _fragment(query, _CARRIER_TERMS), "快递方式", "item_fact")
    if any(term in lowered for term in _AFTER_SALE_TERMS):
        add("service", _fragment(query, _AFTER_SALE_TERMS), "售后或交易规则", "item_fact")
    if re.fullmatch(r"(?:你好|您好|哈喽|hello|hi)[！!。？? ]*", lowered):
        add("service", query, "普通招呼", "greeting")

    repair = next((draft.task for draft in drafts if draft.task.expert == "product" and "维修" in draft.task.normalized_question), None)
    price = next((draft.task for draft in drafts if draft.task.expert == "price"), None)
    if repair is not None and price is not None and "如果" in lowered and any(term in lowered for term in ("我就买", "才买", "才要", "可以吗")):
        index = next(index for index, draft in enumerate(drafts) if draft.task.task_id == price.task_id)
        old = drafts[index].task
        drafts[index] = _Draft(ExpertTask(old.task_id, old.expert, old.question_fragment, old.normalized_question, old.knowledge_scope, old.transaction_conditions, (repair.task_id,)), drafts[index].position, drafts[index].order)
    return drafts


def _requires_model_planning(query: str, context: Mapping[str, object] | None) -> bool:
    lowered = query.casefold()
    return not is_simple_single_question(query) or bool(_context_price_topic(context) and any(term in lowered for term in _FOLLOW_UP_TERMS))


def _call_planner(planner: ExpertPlanner, query: str, history: Sequence[Mapping[str, object]] | None) -> object | None:
    """Treat the optional external model as additive, never as task authority."""

    try:
        return planner(query, history=history)
    except Exception:
        # The injected planner may be a network-backed generator.  Its error
        # must not discard deterministic tasks; retain the traceback for the
        # operator without logging buyer content.
        logger.warning("Xianyu expert planner failed; using rule tasks", exc_info=True)
        return None


def _validated_model_drafts(payload: object | None, query: str) -> list[_Draft]:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return []
    if not isinstance(payload, Mapping) or not isinstance(payload.get("tasks"), list):
        return []
    drafts: list[_Draft] = []
    identifiers: set[str] = set()
    for order, raw in enumerate(payload["tasks"]):
        if not isinstance(raw, Mapping):
            continue
        task_id, expert = raw.get("task_id"), raw.get("expert")
        fragment = raw.get("question_fragment", raw.get("question"))
        normalized = raw.get("normalized_question", raw.get("question"))
        scope = raw.get("knowledge_scope", raw.get("scope"))
        if not (isinstance(task_id, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,31}", task_id) and task_id not in identifiers):
            continue
        if expert not in _VALID_EXPERTS or scope not in _VALID_SCOPES or not _scope_allowed(expert, scope):
            continue
        if not (isinstance(fragment, str) and fragment.strip() and _in_query(fragment, query) and isinstance(normalized, str) and normalized.strip()):
            continue
        conditions = _validate_model_conditions(raw.get("transaction_conditions", raw.get("conditions", {})), query, fragment, expert)
        dependencies = raw.get("depends_on_task_ids", raw.get("depends_on", []))
        if conditions is None or not isinstance(dependencies, list) or not all(isinstance(value, str) for value in dependencies):
            continue
        identifiers.add(task_id)
        task = ExpertTask(task_id, expert, fragment.strip(), normalized.strip(), scope, conditions, tuple(dependencies))  # type: ignore[arg-type]
        drafts.append(_Draft(task, _position(query, fragment), order))
    valid_ids = {draft.task.task_id for draft in drafts}
    normalized_drafts = [
        _Draft(
            ExpertTask(
                draft.task.task_id,
                draft.task.expert,
                draft.task.question_fragment,
                draft.task.normalized_question,
                draft.task.knowledge_scope,
                draft.task.transaction_conditions,
                tuple(dep for dep in draft.task.depends_on_task_ids if dep in valid_ids),
            ),
            draft.position,
            draft.order,
        )
        for draft in drafts
    ]
    if not _requires_repair_dependency(query):
        return normalized_drafts

    repair_task_ids = {
        draft.task.task_id
        for draft in normalized_drafts
        if _is_repair_task(draft.task)
    }
    return [
        draft
        for draft in normalized_drafts
        if draft.task.expert != "price"
        or repair_task_ids.intersection(draft.task.depends_on_task_ids)
    ]


def _merge_drafts(model_drafts: list[_Draft], rule_drafts: list[_Draft], router: IntentRouter) -> list[_Draft]:
    # Rules are the trusted completeness baseline.  A model may add a valid
    # uncovered task, but cannot replace a rule task and accidentally drop its
    # source-derived conditions or dependencies.
    merged, identities = list(rule_drafts), {_task_identity(draft.task, router) for draft in rule_drafts}
    for draft in model_drafts:
        identity = _task_identity(draft.task, router)
        if identity not in identities:
            merged.append(draft)
            identities.add(identity)
    return sorted(merged, key=lambda draft: (draft.position, draft.order))


def _finalize_drafts(drafts: list[_Draft]) -> list[ExpertTask]:
    id_map = {draft.task.task_id: f"q{index}" for index, draft in enumerate(drafts, start=1)}
    return [ExpertTask(id_map[draft.task.task_id], draft.task.expert, draft.task.question_fragment, draft.task.normalized_question, draft.task.knowledge_scope, dict(draft.task.transaction_conditions), tuple(id_map[dep] for dep in draft.task.depends_on_task_ids if dep in id_map)) for draft in drafts]


def _task_identity(task: ExpertTask, router: IntentRouter) -> tuple[str, str, str]:
    if task.expert == "price":
        return task.expert, task.knowledge_scope, str(task.transaction_conditions.get("request_kind", "listed_price"))
    return task.expert, task.knowledge_scope, router.route(task.question_fragment, allow_ai=False).intent


def _scope_allowed(expert: object, scope: object) -> bool:
    return (expert == "price" and scope == "item_fact") or (expert == "product" and scope in {"item_fact", "model_knowledge"}) or (expert == "service" and scope in {"item_fact", "seller_rule", "greeting"})


def _validate_model_conditions(raw: object, query: str, fragment: str, expert: object) -> dict[str, object] | None:
    if not isinstance(raw, Mapping) or (expert != "price" and raw):
        return None
    allowed = {"shipping", "request_kind", "offer_cents", "shipping_comparison", "follow_up"}
    if not set(raw).issubset(allowed):
        return None
    conditions, lowered = dict(raw), (query + " " + fragment).casefold()
    shipping = conditions.get("shipping")
    if shipping == "buyer_pays" and not any(term in lowered for term in _BUYER_PAYS_TERMS):
        return None
    if shipping == "seller_pays" and not _mentions_seller_pays(lowered):
        return None
    if shipping is not None and shipping not in {"buyer_pays", "seller_pays"}:
        return None
    kind = conditions.get("request_kind")
    if kind is not None and kind not in {"minimum", "offer", "additional_discount", "listed_price"}:
        return None
    if kind == "minimum" and "最低" not in lowered:
        return None
    if kind == "additional_discount" and not any(term in lowered for term in ("再少", "再便宜", "再优惠", "再刀")):
        return None
    offers = _offer_cents(query)
    if kind == "offer" and not offers:
        return None
    if "offer_cents" in conditions and conditions["offer_cents"] not in offers:
        return None
    if conditions.get("shipping_comparison") is True and not (_mentions_seller_pays(lowered) and any(term in lowered for term in _BUYER_PAYS_TERMS)):
        return None
    if "shipping_comparison" in conditions and not isinstance(conditions["shipping_comparison"], bool):
        return None
    if "follow_up" in conditions and conditions["follow_up"] is not True:
        return None
    return conditions


def _price_conditions(query: str, context: Mapping[str, object] | None) -> dict[str, object]:
    lowered, conditions = query.casefold(), {}
    buyer_pays = any(term in lowered for term in _BUYER_PAYS_TERMS)
    comparison = buyer_pays and _mentions_seller_pays(lowered)
    if comparison:
        conditions["shipping_comparison"] = True
    elif buyer_pays:
        conditions["shipping"] = "buyer_pays"
    elif "包邮" in lowered:
        conditions["shipping"] = "seller_pays"
    offers = _offer_cents(query)
    if offers and any(term in lowered for term in ("我", "可以", "行吗", "我就买")):
        conditions.update({"request_kind": "offer", "offer_cents": offers[0]})
    elif any(term in lowered for term in ("再少", "再便宜", "再优惠", "再刀")):
        conditions.update({"request_kind": "additional_discount", "follow_up": True})
    elif "最低" in lowered or (buyer_pays and "呢" in lowered and _context_price_topic(context)):
        conditions["request_kind"] = "minimum"
    else:
        conditions["request_kind"] = "listed_price"
    return conditions


def _is_price_question(query: str, context: Mapping[str, object] | None) -> bool:
    lowered = query.casefold()
    if any(term in lowered for term in _BARGAIN_TERMS):
        return True
    if _offer_cents(query) and any(term in lowered for term in ("可以", "行吗", "我就买", "我出")):
        return True
    if any(term in lowered for term in _BUYER_PAYS_TERMS) and ("最低" in lowered or _context_price_topic(context)):
        return True
    return _asks_item_price(query) and not ("包含" in lowered and "价格" in lowered)


def _context_price_topic(context: Mapping[str, object] | None) -> str | None:
    value = context.get("recent_price_topic") if isinstance(context, Mapping) else None
    return value.strip() if isinstance(value, str) and value.strip() else None


def _offer_cents(query: str) -> list[int]:
    cents: list[int] = []
    for match in _OFFER_PATTERN.finditer(query):
        try:
            value = int((Decimal(match.group(1)) * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        except (InvalidOperation, ValueError):
            continue
        if value >= 0 and value not in cents:
            cents.append(value)
    return cents


def _history_question(lowered: str) -> str:
    return "是否有拆修记录" if "拆" in lowered else "是否摔过" if "摔" in lowered or "跌" in lowered else "是否维修过"


def _requires_repair_dependency(query: str) -> bool:
    lowered = query.casefold()
    return (
        "如果" in lowered
        and any(term in lowered for term in _HISTORY_TERMS)
        and any(term in lowered for term in ("我就买", "才买", "才要", "可以吗"))
    )


def _is_repair_task(task: ExpertTask) -> bool:
    text = f"{task.question_fragment} {task.normalized_question}".casefold()
    return task.expert == "product" and any(term in text for term in _HISTORY_TERMS)


def _fragment(query: str, terms: Sequence[str]) -> str:
    positions = [(query.casefold().find(term), term) for term in terms if query.casefold().find(term) >= 0]
    if not positions:
        return query
    start, term = min(positions)
    return _fragment_from_position(query, start, term)


def _price_fragment(query: str) -> str:
    """Keep the price clause in source order, even when punctuation is absent."""

    positions = [
        query.casefold().find(term)
        for term in (*_BARGAIN_TERMS, *_BUYER_PAYS_TERMS, "包邮")
        if query.casefold().find(term) >= 0
    ]
    positions.extend(match.start() for match in _OFFER_PATTERN.finditer(query))
    if not positions:
        return query
    start = min(positions)
    return _fragment_from_position(query, start, query[start:])


def _fragment_from_position(query: str, start: int, fallback: str) -> str:
    punctuation_end = min(
        (index for index in (query.find(mark, start) for mark in "，,。.!！?？、；;") if index >= 0),
        default=len(query),
    )
    boundary_end = min(
        (
            index
            for term in _EXPERT_BOUNDARY_TERMS
            for index in (query.casefold().find(term, start + 1),)
            if index >= 0
        ),
        default=len(query),
    )
    end = min(punctuation_end, boundary_end)
    return query[start:end].strip() or fallback


def _mentions_seller_pays(lowered: str) -> bool:
    """Match an explicit package-included condition, not the substring in 不包邮."""

    return bool(re.search(r"(?<!不)包邮", lowered))


def _position(query: str, fragment: str) -> int:
    found = query.casefold().find(fragment.casefold())
    return found if found >= 0 else len(query)


def _in_query(fragment: str, query: str) -> bool:
    return re.sub(r"\s+", "", fragment).casefold() in re.sub(r"\s+", "", query).casefold()


def is_seller_scoped_query(query: str) -> bool:
    return any(term in str(query or "").casefold() for term in _SELLER_SCOPE_TERMS)


def common_knowledge_query(plan: QuestionPlan, fallback: str) -> str:
    return "；".join(need["question"] for need in plan["knowledge_questions"] if need["scope"] == "common") or fallback
