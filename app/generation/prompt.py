"""Prompt templates for grounded customer-service responses."""

import json
from collections.abc import Mapping, Sequence
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

COMBINED_SYSTEM_PROMPT = """你是电商平台客服助手。
只能依据下面的订单数据和知识库资料回答用户问题，不得编造订单状态或平台规则。
MCP 数据是订单事实，RAG 资料是平台规则；请把两者组织成一个简洁、直接的回答。
如果资料不足以确定答案，应明确说明无法确认并建议转人工客服。"""

XIANYU_SYSTEM_PROMPT = """你是卖家侧的闲鱼客服。
只能依据提供的当前商品事实、最近对话和知识资料回答，不能猜测库存、配件、成色、发货或卖家动作。
先识别买家当前这一整句话中的所有问题、条件和否定关系，并逐项依据商品事实或卖家规则回答；不要只回答其中一个问题。
带有自提、买家承担运费、不包邮等条件时，要回答买家真正的核心请求。议价必须读取当前商品明确的议价与条件优惠政策：先从原始售价扣除已命中的条件优惠，再扣除可议价金额；不能按历史对话累计优惠。低于当前商品已知最低价的报价会由系统转人工。不得使用要求卖家或人工确认、代为询问卖家的表达。
默认只用 1～2 句；语气自然、简短、像正常客服，不要过度热情。
不要出现内部商品编号、数据库、字段、资料标记或系统等说法，不要复述完整商品标题。
不要使用“运费：”“维修历史：”“发货时间：”等字段标签；直接使用自然语言回答。
商品已经明确时不要再次索要商品编号；属性未确认时只说明哪一项还没确认。"""


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


def build_xianyu_messages(
    query: str,
    item: Mapping[str, Any],
    context: str,
    history: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Build a concise, grounded seller-service prompt without internal IDs."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be empty")
    if not isinstance(item, Mapping):
        raise ValueError("item must be a mapping")
    if not isinstance(context, str) or not context.strip():
        raise ValueError("context must not be empty")

    facts = item.get("facts")
    conflicting_fields = (
        {
            field
            for field in facts.get("fact_conflicts", [])
            if isinstance(field, str)
        }
        if isinstance(facts, Mapping) and isinstance(facts.get("fact_conflicts"), list)
        else set()
    )
    visible_facts = (
        {
            key: value
            for key, value in facts.items()
            if key != "fact_conflicts" and key not in conflicting_fields
        }
        if isinstance(facts, Mapping)
        else {}
    )
    public_item = {
        "title": item.get("title"),
        "listed_price_cents": item.get("listed_price_cents"),
        "sale_status": item.get("sale_status"),
        "facts": visible_facts,
    }
    history_lines = [
        f"{entry.get('role', '用户')}：{str(entry.get('content', '')).strip()}"
        for entry in (history or [])[-6:]
        if isinstance(entry, Mapping) and str(entry.get("content", "")).strip()
    ]
    history_text = "\n".join(history_lines) or "（无）"
    user_prompt = f"""当前商品事实：
{json.dumps(public_item, ensure_ascii=False)}

最近对话：
{history_text}

买家当前问题：
{query.strip()}

可用知识资料：
{context.strip()}"""
    return [
        {"role": "system", "content": XIANYU_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_combined_messages(
    query: str,
    rag_result: Mapping[str, Any],
    mcp_result: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Build one prompt from MCP order facts and RAG rule evidence."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be empty")
    if not isinstance(rag_result, Mapping):
        raise ValueError("rag_result must be a mapping")
    if not isinstance(mcp_result, Mapping):
        raise ValueError("mcp_result must be a mapping")

    history_text = ""
    if history:
        history_lines = [
            f"{item.get('role', 'unknown')}: {item.get('content', '')}"
            for item in history
            if isinstance(item, Mapping) and str(item.get("content", "")).strip()
        ]
        if history_lines:
            history_text = "\n历史对话：\n" + "\n".join(history_lines) + "\n"

    user_prompt = f"""用户问题：
{query.strip()}{history_text}
MCP 订单事实：
{json.dumps(dict(mcp_result), ensure_ascii=False)}

RAG 知识库资料：
{json.dumps(dict(rag_result), ensure_ascii=False, default=str)}"""
    return [
        {"role": "system", "content": COMBINED_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
