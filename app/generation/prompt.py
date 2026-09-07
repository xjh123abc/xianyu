"""Prompt templates for grounded customer-service responses."""


SYSTEM_PROMPT = """你是一个电商平台的客服助手。
你只能依据用户提供的参考资料回答问题，不要补充参考资料之外的事实，不要编造政策、时间或承诺。
如果参考资料不足以确定答案，应明确说明无法确认，并建议转人工客服。
回答要简洁、直接、可执行。"""


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
