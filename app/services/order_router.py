"""Rule-based routing for order queries and order-policy follow-ups."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal


Route = Literal["rag", "order", "rag_mcp", "missing_order_id", "unsupported_action"]
RouteResult = tuple[Route, str | None]


_ORDER_ID_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])TEST\d{4}(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_EXPLICIT_ORDER_LOOKUP_PATTERN = re.compile(
    r"(?:帮我|请|麻烦)?\s*"
    r"(?:查|查询|查看)\s*(?:一下|下)?\s*"
    r"(?:我的|这笔|这个)?\s*(?:订单|物流|运单|快递)"
)
_UNSUPPORTED_ACTION_PATTERNS = (
    re.compile(
        r"(?:帮我|请|我要|我想|想要|申请|办理|发起|执行|直接)?\s*"
        r"(?:取消|撤销|删除|关闭)(?:一下)?\s*订单"
    ),
    re.compile(
        r"(?:帮我|请|我要|我想|想要|申请|办理|发起|执行|直接)\s*"
        r"(?:申请)?\s*(?:退款|退货|退货退款)"
    ),
    re.compile(
        r"(?:帮我|请|我要|我想|想要|申请|办理|发起|执行|直接)?\s*"
        r"(?:修改|更改|变更|更换|改)(?:收货)?地址"
    ),
)
_ORDER_QUERY_TERMS = (
    "状态",
    "订单状态",
    "订单信息",
    "订单详情",
    "物流",
    "运单",
    "快递",
    "发货",
    "签收",
    "到货",
)
_PERSONAL_ORDER_TERMS = ("我的订单", "这笔订单", "这个订单", "这单", "我的物流")
_COMBINED_RULE_TERMS = ("一般", "通常", "规则", "政策", "多久", "多长时间", "时效")


def route_query(
    query: str,
    history: Sequence[Mapping[str, Any]] | None = None,
    session_state: Mapping[str, Any] | None = None,
) -> RouteResult:
    """Route a query to RAG, MCP, or the minimal combined workflow."""

    normalized_query = query.strip() if isinstance(query, str) else ""
    if _is_unsupported_action(normalized_query):
        return "unsupported_action", None

    current_order_id = _extract_order_id(normalized_query)
    remembered_order_id = _session_order_id(session_state) or _history_order_id(history)
    order_id = current_order_id or (
        remembered_order_id
        if _is_contextual_followup(normalized_query, remembered_order_id)
        else None
    )
    if _is_combined_query(normalized_query, order_id, remembered_order_id):
        return "rag_mcp", order_id
    if current_order_id is None and _is_rule_followup(normalized_query, remembered_order_id):
        return "rag", None
    if _is_order_query(normalized_query, order_id):
        return ("order", order_id) if order_id is not None else ("missing_order_id", None)
    return "rag", None


def is_rule_followup(query: str, remembered_order_id: str | None) -> bool:
    """Return whether a policy-only follow-up should stay on the RAG path."""

    return _is_rule_followup(query, remembered_order_id)


def session_order_id(session_state: Mapping[str, Any] | None) -> str | None:
    """Read the remembered order ID from one chat session."""

    return _session_order_id(session_state)


def history_order_id(history: Sequence[Mapping[str, Any]] | None) -> str | None:
    """Read the most recent order ID mentioned in chat history."""

    return _history_order_id(history)


def _extract_order_id(query: str) -> str | None:
    match = _ORDER_ID_PATTERN.search(query)
    return match.group(0).upper() if match else None


def _is_unsupported_action(query: str) -> bool:
    return any(pattern.search(query) for pattern in _UNSUPPORTED_ACTION_PATTERNS)


def _is_order_query(query: str, order_id: str | None) -> bool:
    if _EXPLICIT_ORDER_LOOKUP_PATTERN.search(query):
        return True
    if order_id is not None and any(term in query for term in ("查", "查询", "查看")):
        return True
    return any(term in query for term in _ORDER_QUERY_TERMS) and (
        order_id is not None or any(term in query for term in _PERSONAL_ORDER_TERMS)
    )


def _session_order_id(session_state: Mapping[str, Any] | None) -> str | None:
    if not isinstance(session_state, Mapping):
        return None
    order_id = session_state.get("order_id")
    return str(order_id).strip().upper() if order_id else None


def _history_order_id(history: Sequence[Mapping[str, Any]] | None) -> str | None:
    if not history:
        return None
    for item in reversed(history):
        if isinstance(item, Mapping):
            order_id = _extract_order_id(str(item.get("content", "")))
            if order_id:
                return order_id
    return None


def _is_contextual_followup(query: str, session_order_id: str | None) -> bool:
    return session_order_id is not None and any(
        term in query for term in ("它", "这个订单", "这笔", "那", "现在", "当前", "超时")
    )


def _is_combined_query(
    query: str, order_id: str | None, session_order_id: str | None
) -> bool:
    if order_id is None:
        return False
    asks_for_rule = any(term in query for term in _COMBINED_RULE_TERMS)
    asks_for_current_order = any(
        term in query for term in ("现在", "当前", "状态", "已经", "是否", "超时")
    )
    return asks_for_rule and asks_for_current_order or (
        session_order_id is not None and "超时" in query
    )


def _is_rule_followup(query: str, remembered_order_id: str | None) -> bool:
    return remembered_order_id is not None and any(
        term in query for term in _COMBINED_RULE_TERMS
    ) and not any(
        term in query for term in ("现在", "当前", "状态", "已经", "是否", "超时")
    )
