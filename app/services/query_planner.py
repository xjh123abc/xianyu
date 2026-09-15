"""Minimal evidence planning for unified item and seller-knowledge questions."""

from __future__ import annotations

import re
from typing import Literal, TypedDict


ItemField = Literal[
    "listed_price_cents",
    "sale_status",
    "lens",
    "included_items",
    "condition",
    "history",
    "identity",
]
KnowledgeScope = Literal["common", "item"]


class KnowledgeQuestion(TypedDict):
    """One knowledge need and the narrowest corpus scope that may answer it."""

    question: str
    scope: KnowledgeScope


class QuestionPlan(TypedDict):
    """Minimal internal plan for item facts and knowledge evidence."""

    item_fields: list[ItemField]
    knowledge_questions: list[KnowledgeQuestion]


_PRICE_TERMS = ("价格", "多少钱", "标价", "售价", "多少元", "price", "cost")
_SHIPPING_PRICE_TERMS = ("运费", "邮费", "快递费", "shipping fee")
_NON_ITEM_STATUS_TERMS = ("订单状态", "物流状态", "快递状态", "发货状态")
_SELLER_SCOPE_TERMS = ("你们店", "店里", "本店", "卖家", "这件商品", "这个商品")
_STATUS_TERMS = (
    "在吗",
    "能买吗",
    "还能买",
    "可买吗",
    "还有吗",
    "在售",
    "还没卖",
    "没卖",
    "还没出",
    "卖出",
    "售出",
    "已售",
    "状态",
    "available",
    "sold",
)

COMMON_KNOWLEDGE_RETRIEVAL_HINT = (
    "卖家通用规则 售后与边界 二手商品 成色 瑕疵 配件 "
    "双方确认 争议 附加条件 单独确认 不自行承诺"
)
_COMMON_KNOWLEDGE_TERMS = (
    "售后",
    "质量问题",
    "退货",
    "退款",
    "发货",
    "运费",
    "邮费",
    "包邮",
    "规则",
    "政策",
    "shipping",
)
_ITEM_KNOWLEDGE_TERMS = (
    "配件",
    "包含",
    "附带",
    "成色",
    "瑕疵",
    "磕碰",
    "功能",
    "检测",
    "维修",
    "拆修",
    "改装",
    "使用",
    "续航",
    "说明",
    "accessory",
    "condition",
    "repair",
)
_ITEM_REFERENCE_TERMS = (
    "这个商品",
    "这件商品",
    "这个东西",
    "这件",
    "这台",
    "它",
)
_LENS_TERMS = ("镜头", "焦段", "焦距", "lens")
_INCLUDED_ITEMS_TERMS = ("配件", "包含", "附带", "带什么", "一起出", "赠送", "accessory")
_CONDITION_TERMS = ("成色", "外观", "瑕疵", "快门", "过片", "功能", "检测", "condition")
_HISTORY_TERMS = ("维修", "修过", "拆修", "改装", "摔", "磕碰", "维修记录", "repair")
_IDENTITY_TERMS = ("型号", "品牌", "品类", "类别", "model", "brand")


def _asks_item_price(query: str) -> bool:
    """Distinguish the item price from shipping-fee questions."""

    lowered_query = str(query or "").casefold()
    explicitly_item_price = any(
        term in lowered_query for term in ("商品价格", "商品多少钱", "标价", "售价")
    )
    if not explicitly_item_price and any(
        term in lowered_query for term in _SHIPPING_PRICE_TERMS
    ):
        return False
    return any(term in lowered_query for term in _PRICE_TERMS)


def _asks_item_status(query: str) -> bool:
    """Do not turn order or logistics status into an item sale-status lookup."""

    lowered_query = str(query or "").casefold()
    if any(term in lowered_query for term in _NON_ITEM_STATUS_TERMS):
        return False
    return any(term in lowered_query for term in _STATUS_TERMS)


def _structured_item_fields(query: str) -> list[ItemField]:
    """Classify facts that can be answered only from explicit item evidence."""

    lowered_query = str(query or "").casefold()
    fields: list[ItemField] = []
    for field, terms in (
        ("lens", _LENS_TERMS),
        ("included_items", _INCLUDED_ITEMS_TERMS),
        ("condition", _CONDITION_TERMS),
        ("history", _HISTORY_TERMS),
        ("identity", _IDENTITY_TERMS),
    ):
        if any(term in lowered_query for term in terms):
            fields.append(field)
    return fields


def build_question_plan(query: str) -> QuestionPlan:
    """Return the supported fact fields and independently gated knowledge needs."""

    lowered_query = str(query or "").casefold()
    item_fields: list[ItemField] = []
    if _asks_item_price(lowered_query):
        item_fields.append("listed_price_cents")
    if _asks_item_status(lowered_query):
        item_fields.append("sale_status")
    item_fields.extend(_structured_item_fields(lowered_query))

    clauses = [
        clause.strip()
        for clause in re.split(r"[，,。.!！?？、；;]+", str(query or ""))
        if clause.strip()
    ]
    knowledge_questions: list[KnowledgeQuestion] = []
    for clause in clauses:
        lowered_clause = clause.casefold()
        scopes: list[KnowledgeScope] = []
        if any(term in lowered_clause for term in _COMMON_KNOWLEDGE_TERMS):
            scopes.append("common")
        clause_structured_fields = _structured_item_fields(lowered_clause)
        if (
            any(term in lowered_clause for term in _ITEM_KNOWLEDGE_TERMS)
            and not clause_structured_fields
        ):
            scopes.append("item")
        if (
            not scopes
            and not clause_structured_fields
            and not _asks_item_price(lowered_clause)
            and not _asks_item_status(lowered_clause)
            and any(term in lowered_clause for term in _ITEM_REFERENCE_TERMS)
        ):
            scopes.append("item")
        for scope in scopes:
            need = {"question": clause, "scope": scope}
            if need not in knowledge_questions:
                knowledge_questions.append(need)

    return {
        "item_fields": item_fields,
        "knowledge_questions": knowledge_questions,
    }


def is_seller_scoped_query(query: str) -> bool:
    lowered_query = str(query or "").casefold()
    return any(term in lowered_query for term in _SELLER_SCOPE_TERMS)


def common_knowledge_query(plan: QuestionPlan, fallback: str) -> str:
    questions = [
        need["question"]
        for need in plan["knowledge_questions"]
        if need["scope"] == "common"
    ]
    return "；".join(questions) or fallback
