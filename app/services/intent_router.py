"""Rule-first intent routing for Xianyu buyer messages.

The router deliberately classifies only.  It never decides a price, a discount,
or a seller policy; those decisions belong to the structured-fact responder.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast


Intent = Literal[
    "AVAILABILITY",
    "PRICE",
    "BARGAIN",
    "CONDITION",
    "DEFECT",
    "REPAIR_HISTORY",
    "FUNCTION",
    "ACCESSORIES",
    "SHIPPING_TIME",
    "SHIPPING_FEE",
    "PRODUCT_INFO",
    "AFTER_SALE",
    "GREETING",
    "OTHER",
]

VALID_INTENTS = frozenset(
    {
        "AVAILABILITY", "PRICE", "BARGAIN", "CONDITION", "DEFECT",
        "REPAIR_HISTORY", "FUNCTION", "ACCESSORIES", "SHIPPING_TIME",
        "SHIPPING_FEE", "PRODUCT_INFO", "AFTER_SALE", "GREETING", "OTHER",
    }
)


@dataclass(frozen=True)
class IntentMatch:
    intent: Intent
    required_fields: tuple[str, ...]
    source: Literal["rule", "ai", "fallback"]

    @property
    def requires_item(self) -> bool:
        return self.intent not in {"GREETING", "OTHER"}


Classifier = Callable[[str], str | None]


_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "AVAILABILITY": ("sale_status",),
    "PRICE": ("listed_price_cents",),
    "BARGAIN": ("listed_price_cents", "negotiation", "shipping"),
    "CONDITION": ("condition",),
    "DEFECT": ("condition",),
    "REPAIR_HISTORY": ("history",),
    "FUNCTION": ("function",),
    "ACCESSORIES": ("accessories", "accessory_details"),
    "SHIPPING_TIME": ("shipping",),
    "SHIPPING_FEE": ("shipping",),
    "PRODUCT_INFO": ("identity", "product_info"),
    "AFTER_SALE": ("after_sale",),
    "GREETING": (),
    "OTHER": (),
}


_BUYER_NEED_GROUPS = (
    ("多少钱", "价格", "标价", "什么价", "拍的话"),
    ("还在", "还没卖", "没卖", "还没出", "在售", "卖掉", "卖出", "售出", "已售", "有货"),
    ("配件", "带什么", "包含", "镜头"),
    ("成色", "外观", "划痕", "磕碰"),
    ("维修", "修过", "拆修", "摔过"),
    ("发货", "今天能发", "快递", "顺丰", "包邮", "运费", "邮费"),
    ("便宜", "少一点", "优惠", "议价", "刀吗", "最低"),
)
_CONDITIONAL_REQUEST_TERMS = (
    "不用包邮",
    "不包邮",
    "自己出邮费",
    "我出邮费",
    "承担运费",
    "自提",
    "自取",
    "只买机身",
    "不要镜头",
    "如果",
    "的话",
)
_SHIPPING_DISPATCH_TERMS = ("今天能发", "什么时候发", "多久发", "发货")
_SHIPPING_CARRIER_TERMS = ("快递", "顺丰", "中通", "圆通", "韵达", "京东")


def is_simple_single_question(query: str) -> bool:
    """Whether a rule may finish this buyer turn without losing another need.

    Rule replies are intentionally limited to one uncomplicated buyer need.
    A multi-part or conditional message is routed to the existing grounded LLM
    path, which receives the full message and confirmed item facts.
    """

    normalized = str(query or "").strip().casefold()
    if not normalized:
        return False
    if sum(normalized.count(mark) for mark in ("?", "？")) > 1:
        return False
    if any(term in normalized for term in _CONDITIONAL_REQUEST_TERMS):
        return False
    if sum(any(term in normalized for term in group) for group in _BUYER_NEED_GROUPS) > 1:
        return False
    has_dispatch = any(term in normalized for term in _SHIPPING_DISPATCH_TERMS)
    has_carrier = any(term in normalized for term in _SHIPPING_CARRIER_TERMS)
    return not (has_dispatch and has_carrier)


class IntentRouter:
    """Classify buyer messages without deciding prices or seller policy."""

    def __init__(self, classifier: Classifier | None = None) -> None:
        self._classifier = classifier

    def route(self, query: str, *, allow_ai: bool = True) -> IntentMatch:
        """Classify one question, optionally disabling the model fallback.

        Expert planning performs its own single model call for a full compound
        message.  Its local pre-checks therefore set ``allow_ai=False`` so one
        sub-question cannot trigger an additional classification request.
        """

        normalized = str(query or "").strip()
        rule_intent = self._rule_intent(normalized)
        if rule_intent is not None:
            return self._match(rule_intent, "rule")

        if allow_ai and self._classifier is not None and normalized:
            try:
                candidate = self._classifier(normalized)
            except Exception:
                candidate = None
            if isinstance(candidate, str) and candidate.upper() in VALID_INTENTS:
                return self._match(candidate.upper(), "ai")
        return self._match("OTHER", "fallback")

    @staticmethod
    def _match(intent: str, source: Literal["rule", "ai", "fallback"]) -> IntentMatch:
        return IntentMatch(
            intent=cast(Intent, intent),
            required_fields=_REQUIRED_FIELDS[intent],
            source=source,
        )

    @staticmethod
    def _rule_intent(query: str) -> Intent | None:
        lowered = query.casefold()

        # Narrower categories intentionally precede broader ones.
        if any(term in lowered for term in ("闲鱼交易", "可以退", "退货", "退款", "收到发现", "描述一样", "保证", "确认收货", "验货")):
            return "AFTER_SALE"
        if any(term in lowered for term in ("最低", "便宜", "少一点", "少点", "刀吗", "优惠", "出价", "报价", "还价")) or re.search(
            r"我\s*(?:出|给)\s*[¥￥]?\s*\d"
            r"|(?<!\d)[¥￥]?\s*\d+(?:\.\d{1,2})?\s*(?:元|块)?"
            r"\s*(?:包邮|不包邮|自提|自取|面交)?\s*(?:可以|行吗|能出|能收|卖吗)",
            lowered,
        ):
            return "BARGAIN"
        if any(term in lowered for term in ("包邮", "运费", "邮费", "快递费")):
            return "SHIPPING_FEE"
        if any(term in lowered for term in ("什么时候发", "什么时候能发", "今天能发", "从哪里发", "什么快递", "走顺丰", "顺丰", "多久发", "发货")):
            return "SHIPPING_TIME"
        if any(term in lowered for term in ("维修", "修过", "拆修", "拆过", "摔过", "跌过", "维修记录")):
            return "REPAIR_HISTORY"
        if any(term in lowered for term in ("快门", "功能", "正常使用", "直接用", "坏的", "坏了")):
            return "FUNCTION"
        if any(term in lowered for term in ("划痕", "磕碰", "毛病", "隐藏问题", "瑕疵", "问题吗")):
            return "DEFECT"
        if any(term in lowered for term in ("成色", "外观", "新不新", "使用痕迹")):
            return "CONDITION"
        if any(term in lowered for term in ("配件", "带什么", "包含", "一起给", "说明书", "包装", "镜头")):
            return "ACCESSORIES"
        if any(term in lowered for term in ("型号", "哪一年", "哪年生产", "新手", "怎么用", "为什么卖", "为什么要卖")):
            return "PRODUCT_INFO"
        if any(term in lowered for term in ("多少钱", "什么价", "标价", "价格", "拍的话")):
            return "PRICE"
        if any(term in lowered for term in ("还在吗", "还没出", "还没卖", "没卖", "还能拍", "卖掉了吗", "有货", "在售", "卖出")):
            return "AVAILABILITY"
        if re.fullmatch(r"(?:你好|您好|哈喽|hello|hi)[！!。？? ]*", lowered):
            return "GREETING"
        return None
