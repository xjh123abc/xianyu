"""Prompt templates for grounded customer-service responses."""

import json
from collections.abc import Mapping
from typing import Any


SYSTEM_PROMPT = """你是一个电商平台的客服助手。
你只能依据用户提供的参考资料回答问题，不要补充参考资料之外的事实，不要编造政策、时间或承诺。
如果参考资料不足以确定答案，应明确说明无法确认，并建议转人工客服。
回答要简洁、直接、可执行。"""

ORDER_SYSTEM_PROMPT = """你是订单查询助手。下面的数据来自本地模拟订单查询工具。
只根据提供的订单字段回答，不补充未提供的到货时间、地址、金额等信息。
订单号、订单状态、物流状态和运单号必须与数据一致。
运单号为空时说明暂无运单号，不输出字面量 null。
回答中明确标识“本地模拟订单数据”。"""


def build_messages(query: str, context: str) -> list[dict[str, str]]:
    """Build the query/context messages sent to DeepSeek."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be empty")
    if not isinstance(context, str) or not context.strip():
        raise ValueError("context must not be empty")

    user_prompt = f"""用户问题：
{query.strip()}

参考资料：
{context.strip()}"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_order_messages(
    query: str,
    order_data: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Build a grounded prompt from structured MCP order data."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be empty")
    if not isinstance(order_data, Mapping):
        raise ValueError("order_data must be a mapping")

    order_data_json = json.dumps(dict(order_data), ensure_ascii=False)
    user_prompt = f"""用户问题：
{query.strip()}

订单数据：
{order_data_json}"""
    return [
        {"role": "system", "content": ORDER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
