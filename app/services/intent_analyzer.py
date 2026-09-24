"""Semantic understanding for one Xianyu buyer turn."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any

from app.services.chat_contracts import SessionContext
from app.services.intent_contracts import UnderstandingResult, UserNeed


logger = logging.getLogger(__name__)

SemanticPlanner = Callable[..., object]

_PUNCT_RE = re.compile(r"[，,。.!！?？、；;\r\n]+")
_AMOUNT_RE = re.compile(r"(?<!\d)[¥￥]?\s*(\d{1,7}(?:\.\d{1,2})?)\s*(元|块|rmb)?", re.IGNORECASE)

_REPAIR_TERMS = ("修过", "维修", "拆修", "摔过", "摔", "跌过")
_CONDITION_TERMS = ("成色", "外观", "新不新", "使用痕迹")
_AVAILABILITY_TERMS = ("还在吗", "还在售", "在售吗", "还有吗", "有货", "能拍", "能买吗", "还能买", "还没卖", "卖掉了吗")
_IDENTITY_TERMS = ("什么型号", "型号", "品牌")
_ACCESSORY_TERMS = ("配件", "带哪些", "带什么", "带啥", "包含", "附件", "齐不齐")
_LENS_INFO_TERMS = ("带什么镜头", "带啥镜头", "镜头是什么", "什么镜头", "镜头焦段", "焦距")
_FUNCTION_TERMS = ("快门正常", "快门坏", "功能正常", "能不能正常用", "能正常用")
_PRICE_LISTED_TERMS = ("多少钱", "多少价格", "什么价", "价格", "标价", "售价")
_PRICE_MIN_TERMS = ("最低", "底价", "便宜", "少点", "少一点", "优惠", "小刀", "刀吗", "出多少", "多少能出")
_PRICE_MORE_TERMS = ("再少", "再便宜", "再优惠", "再刀", "还能少")
_PRICE_CONFIRM_TERMS = ("就按", "刚才那个价", "刚才说的", "这个价")
_BUYER_PAYS_TERMS = ("不包邮", "不用包邮", "出邮费", "出运费", "自付运费", "承担运费")
_SELLER_PAYS_TERMS = ("包邮",)
_DISPATCH_TERMS = ("今天能发", "明天能发", "什么时候发", "什么时候能发", "多久发", "几天发", "能发吗", "能不能发")
_AFTER_SALE_TERMS = ("退", "退款", "售后", "咋办", "怎么办", "处理", "无理由")
_DAMAGE_TERMS = ("裂", "裂痕", "碎", "坏", "有问题", "质量问题", "破损", "损坏")
_RETURN_SHIPPING_FEE_TERMS = (
    "运费谁出",
    "邮费谁出",
    "退回去运费",
    "退回去邮费",
    "寄回去的钱",
    "寄回的钱",
    "寄回去费用",
    "寄回费用",
    "寄回去运费",
    "寄回运费",
    "运费算谁",
    "邮费算谁",
    "钱算谁",
)
_VAGUE_TERMS = ("那怎么办", "怎么办", "咋办", "那咋办")
_NEGATED_PRODUCT_TERMS = ("不是问镜头型号", "不是问型号", "不是问镜头参数", "不是问成色")


class IntentAnalyzer:
    """Build semantic UserNeed values before Planner creates executable tasks."""

    def __init__(
        self,
        semantic_planner: SemanticPlanner | None = None,
        *,
        timeout_seconds: float = 8.0,
    ) -> None:
        self._semantic_planner = semantic_planner
        self._timeout_seconds = timeout_seconds

    async def analyze(
        self,
        message: str,
        context: SessionContext,
        *,
        item_id: str | None = None,
    ) -> UnderstandingResult:
        """Return a validated semantic understanding for the current turn."""

        query = str(message or "").strip()
        if not query:
            return UnderstandingResult.error("empty_intent_query")

        deterministic = _deterministic_understanding(query, context, item_id=item_id)
        if deterministic is not None:
            return deterministic

        if self._semantic_planner is None:
            return UnderstandingResult.error("intent_semantic_model_unavailable")

        try:
            payload = await asyncio.wait_for(
                asyncio.to_thread(
                    self._semantic_planner,
                    query,
                    history=context.history,
                    timeout_seconds=self._timeout_seconds,
                ),
                timeout=self._timeout_seconds,
            )
        except (asyncio.TimeoutError, TimeoutError):
            logger.warning("Intent semantic model timed out")
            return UnderstandingResult.error("intent_model_timeout", model_called=True)
        except Exception:
            logger.warning("Intent semantic model failed", exc_info=True)
            return UnderstandingResult.error("intent_model_failed", model_called=True)

        if payload is None:
            return UnderstandingResult.error(
                "intent_semantic_model_unavailable",
                model_called=True,
            )

        parsed = _parse_model_payload(payload, query)
        if parsed.status == "ready":
            return UnderstandingResult.ready(
                parsed.needs,
                dependencies=parsed.dependencies,
                raw_response=payload,
                model_called=True,
            )
        if parsed.status == "clarify":
            return UnderstandingResult.clarify(
                parsed.clarification_question or "请补充你想确认的具体问题。",
                raw_response=payload,
                model_called=True,
            )
        return UnderstandingResult.error(
            parsed.error_reason or "intent_parse_failed",
            raw_response=payload,
            model_called=True,
        )


def _deterministic_understanding(
    query: str,
    context: SessionContext,
    *,
    item_id: str | None,
) -> UnderstandingResult | None:
    lowered = query.casefold()
    if _is_vague_followup(lowered) and not _recent_user_text(context.history):
        return UnderstandingResult.clarify("你想确认哪种情况？比如商品状态、价格、发货还是售后。")

    needs: list[UserNeed] = []

    if _is_greeting(lowered):
        needs.append(
            _need(
                "service.greeting",
                query,
                "普通招呼",
                subject="seller",
                outcome="greeting",
            )
        )
        return UnderstandingResult.ready(needs)

    repair_as_condition = _repair_is_only_transaction_condition(lowered)
    if _has_any(lowered, _AVAILABILITY_TERMS):
        needs.append(
            _need(
                "product.availability",
                _source(query, _AVAILABILITY_TERMS),
                "确认当前商品是否还在售",
                subject="当前商品",
                outcome="sale_status",
            )
        )

    if _has_any(lowered, _IDENTITY_TERMS):
        needs.append(
            _need(
                "product.identity_model",
                _source(query, _IDENTITY_TERMS),
                "查询当前商品品牌或型号",
                subject="当前商品",
                outcome="identity_model",
            )
        )

    if _has_any(lowered, _REPAIR_TERMS) and not repair_as_condition:
        needs.append(
            _need(
                "product.repair_history",
                _source(query, _REPAIR_TERMS),
                "查询当前商品是否有维修或拆修记录",
                subject="当前商品",
                outcome="repair_history",
            )
        )

    if _has_any(lowered, _CONDITION_TERMS) and not _negates_product_condition(lowered):
        needs.append(
            _need(
                "product.condition_summary",
                _source(query, _CONDITION_TERMS),
                "查询当前商品成色和外观情况",
                subject="当前商品",
                outcome="condition_summary",
            )
        )

    if _asks_current_damage(lowered):
        needs.append(
            _need(
                "product.condition_issue",
                _source(query, ("镜头", "快门", *_DAMAGE_TERMS)),
                "确认当前商品是否存在买家询问的损坏或瑕疵",
                subject=_subject_from_text(lowered),
                outcome="current_item_condition",
            )
        )

    if _asks_lens_or_accessories(lowered) and not _negates_lens_info(lowered):
        intent = "product.lens_details" if _has_any(lowered, _LENS_INFO_TERMS) else "product.accessories"
        normalized = (
            "查询当前商品随附镜头信息"
            if intent == "product.lens_details"
            else "查询当前商品包含的配件"
        )
        needs.append(
            _need(
                intent,
                _source(query, (*_LENS_INFO_TERMS, *_ACCESSORY_TERMS)),
                normalized,
                subject="当前商品",
                outcome="product_included_items",
            )
        )

    if _has_any(lowered, _FUNCTION_TERMS) and not _asks_after_sale(lowered):
        needs.append(
            _need(
                "product.function",
                _source(query, _FUNCTION_TERMS),
                "查询当前商品功能是否正常",
                subject="当前商品",
                outcome="function_status",
            )
        )

    if _asks_model_knowledge(lowered):
        needs.append(
            _need(
                "product.model_knowledge",
                query,
                query.rstrip("？?。"),
                subject="商品型号或技术知识",
                outcome="model_knowledge",
            )
        )

    price_needs = _price_needs(query, context, condition_on_repair=repair_as_condition)
    if repair_as_condition and price_needs:
        repair_need = _need(
            "product.repair_history",
            _source(query, _REPAIR_TERMS),
            "核验当前商品是否未维修过",
            subject="当前商品",
            outcome="repair_history",
            reply_required=False,
            need_id="repair_condition",
            conditions=(
                {
                    "type": "transaction_precondition",
                    "fact": "repair_history",
                    "expected": "not_repaired",
                },
            ),
        )
        needs.append(repair_need)
        price_needs = [
            _replace_need_dependencies(need, ("repair_condition",))
            for need in price_needs
        ]
    needs.extend(price_needs)

    if _has_any(lowered, _DISPATCH_TERMS):
        needs.append(
            _need(
                "shipping.dispatch_time",
                _source(query, _DISPATCH_TERMS),
                "确认当前商品什么时候可以发货",
                subject="当前商品",
                outcome="dispatch_time",
            )
        )

    if _asks_shipping_fee(lowered) and not price_needs and not _asks_return_shipping_fee(lowered):
        needs.append(
            _need(
                "shipping.fee",
                _source(query, ("包邮", "运费", "邮费", "快递费")),
                "咨询当前商品运费或是否包邮",
                subject="当前商品",
                outcome="shipping_fee",
            )
        )

    if _asks_carrier(lowered):
        needs.append(
            _need(
                "shipping.carrier",
                _source(query, ("什么快递", "走什么快递", "发什么快递", "顺丰", "中通", "圆通", "韵达", "京东")),
                "咨询当前商品使用什么快递",
                subject="当前商品",
                outcome="shipping_carrier",
            )
        )

    after_sale_needs = _after_sale_needs(query, context)
    needs.extend(after_sale_needs)

    deduped = _dedupe_needs(needs)
    if deduped:
        return UnderstandingResult.ready(_renumber_needs(_ordered_needs(query, deduped)))
    return None


def _parse_model_payload(payload: object, query: str) -> UnderstandingResult:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return UnderstandingResult.error("intent_parse_failed", raw_response=payload)
    if not isinstance(payload, Mapping):
        return UnderstandingResult.error("intent_parse_failed", raw_response=payload)
    status = payload.get("status")
    if status == "clarify":
        question = payload.get("clarification_question")
        return UnderstandingResult.clarify(
            question if isinstance(question, str) and question.strip() else "请补充你想确认的具体问题。",
            raw_response=payload,
        )
    if status == "error":
        reason = payload.get("error_reason")
        return UnderstandingResult.error(
            reason if isinstance(reason, str) and reason.strip() else "intent_parse_failed",
            raw_response=payload,
        )
    if status != "ready" or not isinstance(payload.get("needs"), list):
        return UnderstandingResult.error("intent_parse_failed", raw_response=payload)

    needs: list[UserNeed] = []
    for index, raw in enumerate(payload["needs"], start=1):
        if not isinstance(raw, Mapping):
            return UnderstandingResult.error("intent_parse_failed", raw_response=payload)
        source_texts = raw.get("source_texts")
        original = raw.get("original_question")
        if not isinstance(original, str) or not original.strip():
            return UnderstandingResult.error("intent_parse_failed", raw_response=payload)
        if original.strip() not in query and not _source_texts_in_query(source_texts, query):
            return UnderstandingResult.error("intent_source_text_invalid", raw_response=payload)
        try:
            needs.append(
                UserNeed(
                    need_id=str(raw.get("need_id") or f"n{index}"),
                    intent=str(raw.get("intent") or ""),
                    original_question=original,
                    normalized_question=str(raw.get("normalized_question") or original),
                    subject=raw.get("subject") if isinstance(raw.get("subject"), str) else None,
                    conditions=tuple(
                        condition
                        for condition in raw.get("conditions", [])
                        if isinstance(condition, Mapping)
                    )
                    if isinstance(raw.get("conditions", []), list)
                    else (),
                    source_texts=tuple(
                        source for source in raw.get("source_texts", [original])
                        if isinstance(source, str)
                    )
                    if isinstance(raw.get("source_texts", [original]), list)
                    else (original,),
                    context_references=tuple(
                        reference
                        for reference in raw.get("context_references", [])
                        if isinstance(reference, Mapping)
                    )
                    if isinstance(raw.get("context_references", []), list)
                    else (),
                    requested_outcome=raw.get("requested_outcome")
                    if isinstance(raw.get("requested_outcome"), str)
                    else None,
                    reply_required=raw.get("reply_required", True) is not False,
                    depends_on_need_ids=tuple(
                        dependency
                        for dependency in raw.get("depends_on_need_ids", [])
                        if isinstance(dependency, str)
                    )
                    if isinstance(raw.get("depends_on_need_ids", []), list)
                    else (),
                )
            )
        except ValueError:
            return UnderstandingResult.error("intent_contract_invalid", raw_response=payload)
    return UnderstandingResult.ready(_renumber_needs(needs), raw_response=payload)


def _need(
    intent: str,
    original_question: str,
    normalized_question: str,
    *,
    subject: str | None = None,
    outcome: str | None = None,
    conditions: Sequence[Mapping[str, object]] = (),
    reply_required: bool = True,
    depends_on: Sequence[str] = (),
    context_references: Sequence[Mapping[str, object]] = (),
    need_id: str = "pending",
) -> UserNeed:
    return UserNeed(
        need_id=need_id,
        intent=intent,
        original_question=original_question.strip("，,。.!！?？、；; ") or original_question,
        normalized_question=normalized_question.strip("，,。.!！?？、；; ") or normalized_question,
        subject=subject,
        conditions=tuple(dict(condition) for condition in conditions),
        source_texts=(original_question.strip(),),
        context_references=tuple(dict(reference) for reference in context_references),
        requested_outcome=outcome,
        reply_required=reply_required,
        depends_on_need_ids=tuple(depends_on),
    )


def _replace_need_dependencies(need: UserNeed, dependencies: Sequence[str]) -> UserNeed:
    return UserNeed(
        need.need_id,
        need.intent,
        need.original_question,
        need.normalized_question,
        subject=need.subject,
        conditions=need.conditions,
        source_texts=need.source_texts,
        context_references=need.context_references,
        requested_outcome=need.requested_outcome,
        reply_required=need.reply_required,
        depends_on_need_ids=tuple(dependencies),
    )


def _renumber_needs(needs: Sequence[UserNeed]) -> tuple[UserNeed, ...]:
    id_map = {need.need_id: f"n{index}" for index, need in enumerate(needs, start=1)}
    numbered: list[UserNeed] = []
    for index, need in enumerate(needs, start=1):
        numbered.append(
            UserNeed(
                f"n{index}",
                need.intent,
                need.original_question,
                need.normalized_question,
                subject=need.subject,
                conditions=need.conditions,
                source_texts=need.source_texts,
                context_references=need.context_references,
                requested_outcome=need.requested_outcome,
                reply_required=need.reply_required,
                depends_on_need_ids=tuple(
                    id_map[dependency]
                    for dependency in need.depends_on_need_ids
                    if dependency in id_map
                ),
            )
        )
    return tuple(numbered)


def _dedupe_needs(needs: Sequence[UserNeed]) -> list[UserNeed]:
    deduped: list[UserNeed] = []
    seen: set[tuple[str, str, str]] = set()
    for need in needs:
        identity = (need.intent, need.original_question, need.normalized_question)
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(need)
    return deduped


def _ordered_needs(query: str, needs: Sequence[UserNeed]) -> list[UserNeed]:
    indexed = list(enumerate(needs))
    indexed.sort(key=lambda entry: (_need_position(query, entry[1]), entry[0]))
    return [need for _index, need in indexed]


def _need_position(query: str, need: UserNeed) -> int:
    lowered_query = query.casefold()
    candidates = [*need.source_texts, need.original_question]
    for candidate in candidates:
        text = str(candidate or "").strip().casefold()
        if not text:
            continue
        position = lowered_query.find(text)
        if position >= 0:
            return position
    return len(query)


def _price_needs(
    query: str,
    context: SessionContext,
    *,
    condition_on_repair: bool,
) -> list[UserNeed]:
    lowered = query.casefold()
    needs: list[UserNeed] = []
    amounts = _offer_cents(query)
    buyer_pays = _has_any(lowered, _BUYER_PAYS_TERMS)
    seller_pays = _mentions_seller_pays(lowered)
    shipping_condition = "buyer_pays" if buyer_pays else "seller_pays" if seller_pays else None
    conditions: list[dict[str, object]] = []
    if shipping_condition is not None:
        conditions.append({"type": "shipping", "value": shipping_condition})

    xianyu_context = context.platform_context.get("xianyu", {})
    if _has_any(lowered, _PRICE_CONFIRM_TERMS) and (
        xianyu_context.get("recent_price_topic")
        or context.negotiation.get("last_ai_offer") is not None
    ):
        needs.append(
            _need(
                "price.confirm",
                _source(query, _PRICE_CONFIRM_TERMS),
                "确认沿用上一轮报价",
                subject="当前商品",
                outcome="price_confirmation",
                conditions=conditions,
            )
        )
        return needs

    if amounts and _is_offer_context(lowered):
        amount = amounts[0]
        offer_conditions = [*conditions, {"type": "amount", "offer_cents": amount}]
        if condition_on_repair:
            offer_conditions.append(
                {
                    "type": "transaction_precondition",
                    "fact": "repair_history",
                    "expected": "not_repaired",
                }
            )
        needs.append(
            _need(
                "price.offer",
                _source(query, ("我出", "我就买", "可以", "行不", "能出", "能卖")),
                "判断买家报价是否可以接受",
                subject="当前商品",
                outcome="price_decision",
                conditions=offer_conditions,
            )
        )

    asks_more = _has_any(lowered, _PRICE_MORE_TERMS)
    min_source = _source(query, _PRICE_MIN_TERMS)
    more_source = _source(query, _PRICE_MORE_TERMS) if asks_more else ""
    if _has_any(lowered, _PRICE_MIN_TERMS) and not (
        asks_more and min_source == more_source
    ):
        needs.append(
            _need(
                "price.minimum",
                min_source,
                "查询当前商品最低可成交价格",
                subject="当前商品",
                outcome="minimum_price",
                conditions=conditions,
            )
        )
    elif _has_any(lowered, _PRICE_LISTED_TERMS):
        needs.append(
            _need(
                "price.listed_price",
                _source(query, _PRICE_LISTED_TERMS),
                "查询当前商品标价或售价",
                subject="当前商品",
                outcome="listed_price",
                conditions=conditions,
            )
        )
    if asks_more:
        needs.append(
            _need(
                "price.additional_discount",
                more_source,
                "询问当前条件下是否还能继续优惠",
                subject="当前商品",
                outcome="additional_discount",
                conditions=conditions,
            )
        )
    if not needs and buyer_pays and "呢" in lowered and xianyu_context.get("recent_price_topic"):
        needs.append(
            _need(
                "price.minimum",
                _source(query, _BUYER_PAYS_TERMS),
                "结合上一轮价格话题，查询不包邮条件下最低价格",
                subject="当前商品",
                outcome="minimum_price",
                conditions=conditions,
            )
        )
    return needs


def _after_sale_needs(query: str, context: SessionContext) -> list[UserNeed]:
    lowered = query.casefold()
    needs: list[UserNeed] = []
    condition = _scenario_condition(lowered)
    if _asks_after_sale(lowered) and _asks_return_or_after_sale_policy(lowered):
        normalized = _after_sale_normalized(condition, lowered)
        references = ()
        if _is_contextual_return_followup(lowered):
            previous = _recent_user_text(context.history)
            if previous:
                references = ({"role": "user", "content": previous},)
                normalized = f"结合上一轮情景，{normalized}"
        needs.append(
            _need(
                "after_sale.consult",
                _after_sale_source(query),
                normalized,
                subject=_subject_from_text(lowered),
                outcome="return_or_after_sale_policy",
                conditions=(condition,) if condition else (),
                context_references=references,
            )
        )
    if _asks_return_shipping_fee(lowered):
        needs.append(
            _need(
                "after_sale.return_shipping_fee",
                _source(query, _RETURN_SHIPPING_FEE_TERMS),
                _return_shipping_fee_normalized(condition),
                subject="退货运费",
                outcome="return_shipping_fee",
                conditions=(condition,) if _has_scenario_context(condition) else (),
            )
        )
    if "多久到" in lowered or "多久到账" in lowered:
        needs.append(
            _need(
                "after_sale.refund_timing",
                _source(query, ("多久到", "多久到账")),
                _refund_timing_normalized(condition),
                subject="退款时效",
                outcome="refund_timing",
                conditions=(condition,) if _has_scenario_context(condition) else (),
            )
        )
    return needs


def _source(query: str, terms: Sequence[str]) -> str:
    parts = [part.strip() for part in _PUNCT_RE.split(query) if part.strip()]
    for part in parts:
        lowered = part.casefold()
        if any(term and term in lowered for term in terms):
            return part
    return query.strip()


def _after_sale_source(query: str) -> str:
    parts = [part.strip() for part in _PUNCT_RE.split(query) if part.strip()]
    if not parts:
        return query.strip()
    for part in parts:
        lowered = part.casefold()
        if (
            any(term in lowered for term in _AFTER_SALE_TERMS)
            and (
                any(term in lowered for term in _DAMAGE_TERMS)
                or "无理由" in lowered
                or "售后" in lowered
            )
        ):
            return part
    damage_positions = [
        index
        for index, part in enumerate(parts)
        if any(term in part.casefold() for term in _DAMAGE_TERMS)
    ]
    policy_positions = [
        index
        for index, part in enumerate(parts)
        if _asks_return_or_after_sale_policy(part.casefold())
    ]
    if damage_positions and policy_positions:
        first_policy = policy_positions[0]
        context_positions = [
            position for position in damage_positions if position <= first_policy
        ]
        if not context_positions:
            context_positions = [damage_positions[0]]
        selected = sorted({*context_positions, first_policy})
        return "，".join(parts[index] for index in selected).strip("，,。.!！?？、；; ")
    after_sale_positions = [
        index
        for index, part in enumerate(parts)
        if any(term in part.casefold() for term in _AFTER_SALE_TERMS)
    ]
    if damage_positions and after_sale_positions:
        return query.strip("，,。.!！?？、；; ")
    return _source(query, (*_AFTER_SALE_TERMS, *_DAMAGE_TERMS, "7天", "无理由"))


def _has_any(lowered: str, terms: Sequence[str]) -> bool:
    return any(term in lowered for term in terms)


def _is_greeting(lowered: str) -> bool:
    return re.fullmatch(r"(?:你好|您好|哈喽|hello|hi)[！!。？? ]*", lowered) is not None


def _is_vague_followup(lowered: str) -> bool:
    return lowered.strip("？?。!！ ") in _VAGUE_TERMS


def _recent_user_text(history: Sequence[Mapping[str, object]]) -> str | None:
    for entry in reversed(history):
        if not isinstance(entry, Mapping):
            continue
        role = str(entry.get("role", "")).casefold()
        content = str(entry.get("content", "")).strip()
        if content and role in {"user", "buyer", "用户", "买家"}:
            return content
    return None


def _repair_is_only_transaction_condition(lowered: str) -> bool:
    return (
        "如果" in lowered
        and any(term in lowered for term in ("没修过", "没有修过", "无维修"))
        and any(term in lowered for term in ("我就买", "才买", "才要"))
    )


def _negates_lens_info(lowered: str) -> bool:
    return any(term in lowered for term in ("不是问镜头型号", "不是问镜头参数", "不用问镜头型号"))


def _negates_product_condition(lowered: str) -> bool:
    return any(term in lowered for term in _NEGATED_PRODUCT_TERMS)


def _asks_current_damage(lowered: str) -> bool:
    return (
        any(term in lowered for term in _DAMAGE_TERMS)
        and any(term in lowered for term in ("是不是", "有没有", "是否"))
        and not _asks_after_sale(lowered)
    )


def _asks_lens_or_accessories(lowered: str) -> bool:
    return _has_any(lowered, _LENS_INFO_TERMS) or _has_any(lowered, _ACCESSORY_TERMS)


def _asks_model_knowledge(lowered: str) -> bool:
    if (
        _asks_after_sale(lowered)
        or _asks_lens_or_accessories(lowered)
        or _asks_shipping_fee(lowered)
        or _asks_carrier(lowered)
    ):
        return False
    return any(
        term in lowered
        for term in ("50mm", "适合", "怎么用", "怎么装", "安装", "测光", "电池", "兼容", "外接", "闪光灯", "自拍功能", "数据线", "usb")
    )


def _asks_after_sale(lowered: str) -> bool:
    if "不用退货" in lowered or "不是退货" in lowered:
        return False
    return (
        _has_any(lowered, _AFTER_SALE_TERMS)
        and (
            _has_any(lowered, _DAMAGE_TERMS)
            or "无理由" in lowered
            or "售后" in lowered
            or "退" in lowered
            or "有问题" in lowered
        )
    )


def _asks_return_or_after_sale_policy(lowered: str) -> bool:
    return any(
        term in lowered
        for term in (
            "可以退",
            "能退",
            "能不能退",
            "可不可以退",
            "给退",
            "支持退",
            "退吗",
            "退不",
            "退货吗",
            "请直接退款",
            "直接退款",
            "售后怎么",
            "怎么处理",
            "咋办",
            "怎么办",
            "无理由",
        )
    )


def _asks_shipping_fee(lowered: str) -> bool:
    return any(term in lowered for term in ("包邮", "运费", "邮费", "快递费"))


def _asks_return_shipping_fee(lowered: str) -> bool:
    return any(term in lowered for term in _RETURN_SHIPPING_FEE_TERMS)


def _asks_carrier(lowered: str) -> bool:
    return any(term in lowered for term in ("什么快递", "走什么快递", "发什么快递", "顺丰", "中通", "圆通", "韵达", "京东"))


def _is_contextual_return_followup(lowered: str) -> bool:
    return lowered.strip("？?。!！ ") in {"能退吗", "能退不", "可以退吗", "给退不"}


def _scenario_condition(lowered: str) -> dict[str, object]:
    has_hypothetical_marker = any(term in lowered for term in ("如果", "要是", "假如"))
    modality = "hypothetical" if has_hypothetical_marker else "unspecified"
    if any(term in lowered for term in ("收到", "到手", "收货")) and not has_hypothetical_marker:
        modality = "reported_unverified"
    event = "商品有问题"
    if "镜片" in lowered and "碎" in lowered:
        event = "镜片碎了"
    elif "镜头" in lowered and any(term in lowered for term in ("裂", "裂痕")):
        event = "镜头裂了"
    elif "快门" in lowered and "坏" in lowered:
        event = "快门坏了"
    elif any(term in lowered for term in ("有问题", "坏")):
        event = "收到后有问题"
    elif "无理由" in lowered:
        event = "七天无理由退货"
        modality = "unspecified"
    if (
        not has_hypothetical_marker
        and modality == "unspecified"
        and event not in {"商品有问题", "七天无理由退货"}
    ):
        modality = "reported_unverified"
    timing = "收到商品时" if any(term in lowered for term in ("收到", "到手", "收货")) else "unspecified"
    return {
        "type": "scenario",
        "event": event,
        "timing": timing,
        "modality": modality,
    }


def _after_sale_normalized(condition: Mapping[str, object], lowered: str) -> str:
    event = str(condition.get("event") or "该情况")
    timing = str(condition.get("timing") or "unspecified")
    modality = condition.get("modality")
    if "请直接退款" in lowered:
        return f"买家报告{event}并请求直接退款，确认当前支持的售后处理方式"
    if modality == "reported_unverified":
        timing_text = "收到商品时" if timing == "收到商品时" else "当前"
        return f"买家反馈{timing_text}{event}，在未核实前确认当前支持的售后处理方式"
    if modality == "hypothetical":
        return f"假设{event}，咨询是否支持退货或售后处理"
    return f"针对{event}这一未核实情形，咨询是否支持退货或售后处理"


def _return_shipping_fee_normalized(condition: Mapping[str, object]) -> str:
    event = str(condition.get("event") or "")
    timing = str(condition.get("timing") or "unspecified")
    modality = condition.get("modality")
    if event and event != "商品有问题":
        if modality == "reported_unverified":
            timing_text = "收到商品时" if timing == "收到商品时" else "当前"
            return f"买家反馈{timing_text}{event}，在未核实前咨询退回商品时运费由谁承担"
        if modality == "hypothetical":
            return f"假设{event}，咨询退回商品时运费由谁承担"
        return f"针对{event}这一未核实情形，咨询退回商品时运费由谁承担"
    return "咨询退回商品时运费由谁承担"


def _refund_timing_normalized(condition: Mapping[str, object]) -> str:
    event = str(condition.get("event") or "")
    timing = str(condition.get("timing") or "unspecified")
    modality = condition.get("modality")
    if event and event != "商品有问题":
        if modality == "reported_unverified":
            timing_text = "收到商品时" if timing == "收到商品时" else "当前"
            return f"买家反馈{timing_text}{event}，在未核实前咨询售后退款大概多久处理到账"
        if modality == "hypothetical":
            return f"假设{event}，咨询售后退款大概多久处理到账"
        return f"针对{event}这一未核实情形，咨询售后退款大概多久处理到账"
    return "咨询售后退款大概多久处理到账"


def _has_scenario_context(condition: Mapping[str, object]) -> bool:
    event = condition.get("event")
    return isinstance(event, str) and event and event != "商品有问题"


def _subject_from_text(lowered: str) -> str:
    if "镜片" in lowered:
        return "镜片"
    if "镜头" in lowered:
        return "镜头"
    if "快门" in lowered:
        return "快门"
    return "当前商品"


def _offer_cents(query: str) -> list[int]:
    cents: list[int] = []
    for match in _AMOUNT_RE.finditer(query):
        suffix = query[match.end() : match.end() + 3].casefold()
        if suffix.startswith(("mm", "毫米", "小时", "天", "年")):
            continue
        if not match.group(2) and int(float(match.group(1))) < 100:
            continue
        try:
            value = int((Decimal(match.group(1)) * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        except (InvalidOperation, ValueError):
            continue
        if value not in cents:
            cents.append(value)
    return cents


def _is_offer_context(lowered: str) -> bool:
    return any(term in lowered for term in ("我出", "我就买", "可以吗", "行不", "行吗", "能出", "能卖"))


def _mentions_seller_pays(lowered: str) -> bool:
    return "包邮" in lowered and not any(term in lowered for term in _BUYER_PAYS_TERMS)


def _source_texts_in_query(source_texts: object, query: str) -> bool:
    return (
        isinstance(source_texts, list)
        and bool(source_texts)
        and all(isinstance(source, str) and source.strip() in query for source in source_texts)
    )
