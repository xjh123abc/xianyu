"""DeepSeek answer generation through the OpenAI-compatible API."""

from collections.abc import Mapping, Sequence
import json
import re
import time
from typing import Any

from openai import OpenAI

from app.generation.prompt import (
    build_combined_messages,
    build_messages,
    build_order_messages,
    build_xianyu_messages,
)
from app.generation.xianyu_expert_prompt import (
    build_xianyu_expert_plan_messages,
    build_xianyu_product_expert_messages,
    build_xianyu_service_expert_messages,
)
from config.settings import settings


class DeepSeekGenerator:
    """Generate a grounded answer from a query and ContextBuilder text."""

    _EMPTY_CONTENT_RETRIES = 1
    _RETRY_MAX_TOKENS = 2048

    def __init__(self, client: Any | None = None) -> None:
        """Accept an injected client for tests; otherwise build it lazily."""
        self.client = client

    def generate(self, query: str, context: str) -> str:
        """Send query and context to DeepSeek and return its answer text."""
        return self._generate_messages(build_messages(query, context))

    def generate_order(
        self,
        query: str,
        order_data: Mapping[str, Any],
    ) -> str:
        """Use the existing DeepSeek client flow with an order-specific prompt."""

        return self._generate_messages(build_order_messages(query, order_data))

    def generate_xianyu(
        self,
        query: str,
        item: Mapping[str, Any],
        context: str,
        *,
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> str:
        """Generate a natural, evidence-bound reply for a seller conversation."""

        return self._generate_messages(
            build_xianyu_messages(query, item, context, history)
        )

    def plan_xianyu_questions(
        self,
        query: str,
        *,
        history: Sequence[Mapping[str, Any]] | None = None,
        timeout_seconds: float | None = None,
    ) -> str:
        """Return a planner payload; S4 validates it before creating tasks."""

        return self._generate_messages(
            build_xianyu_expert_plan_messages(query, history),
            timeout_seconds=timeout_seconds,
        )

    def generate_xianyu_expert(
        self,
        expert: str,
        question: str,
        item: Mapping[str, Any] | None,
        evidence: str,
        *,
        history: Sequence[Mapping[str, Any]] | None = None,
        timeout_seconds: float | None = None,
    ) -> str:
        """Generate one product or service answer from its bounded evidence."""

        if expert == "product":
            messages = build_xianyu_product_expert_messages(
                question, item, evidence, history
            )
        elif expert == "service":
            messages = build_xianyu_service_expert_messages(
                question, item, evidence, history
            )
        else:
            raise ValueError("expert must be 'product' or 'service'")
        return self._generate_messages(messages, timeout_seconds=timeout_seconds)

    def generate_combined(
        self,
        query: str,
        rag_result: Mapping[str, Any],
        mcp_result: Mapping[str, Any],
        *,
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> str:
        """Generate the final answer from both independent service results."""
        return self._generate_messages(
            build_combined_messages(query, rag_result, mcp_result, history)
        )

    def classify_intent(self, query: str) -> str | None:
        """Return one routing label from a deliberately decision-free prompt.

        This is only the fallback after local rules.  The model never receives
        seller facts and therefore cannot choose a price, discount, or policy.
        """

        content = self._generate_messages(
            [
                {
                    "role": "system",
                    "content": (
                        "你是闲鱼买家问题分类器。只输出 JSON 对象，格式为 "
                        '{"intent":"..."}。intent 必须是以下之一：'
                        "AVAILABILITY, PRICE, BARGAIN, CONDITION, DEFECT, "
                        "REPAIR_HISTORY, FUNCTION, ACCESSORIES, SHIPPING_TIME, "
                        "SHIPPING_FEE, PRODUCT_INFO, AFTER_SALE, GREETING, OTHER。"
                        "不要回答买家，不要添加事实或解释。"
                    ),
                },
                {"role": "user", "content": query},
            ]
        )
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r'"intent"\s*:\s*"([A-Z_]+)"', content.upper())
            return match.group(1) if match else None
        intent = payload.get("intent") if isinstance(payload, dict) else None
        return intent.strip().upper() if isinstance(intent, str) else None

    def _generate_messages(
        self,
        messages: list[dict[str, str]],
        *,
        timeout_seconds: float | None = None,
    ) -> str:
        if settings is None:
            raise RuntimeError("Project settings are unavailable")

        client = self.client or self._build_client()
        request: dict[str, object] = {
            "model": settings.deepseek_model,
            "messages": messages,
            "temperature": settings.deepseek_temperature,
            "max_tokens": settings.deepseek_max_tokens,
            "stream": False,
        }
        if settings.deepseek_model == "deepseek-flash":
            request["extra_body"] = {"thinking": {"type": "disabled"}}
        deadline = (
            time.monotonic() + max(float(timeout_seconds), 0.01)
            if timeout_seconds is not None
            else None
        )

        for attempt in range(self._EMPTY_CONTENT_RETRIES + 1):
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                request["timeout"] = max(remaining, 0.01)

            response = client.chat.completions.create(**request)
            try:
                content = response.choices[0].message.content
            except (AttributeError, IndexError, TypeError) as error:
                raise RuntimeError(
                    "DeepSeek response did not contain an answer"
                ) from error

            if isinstance(content, str) and content.strip():
                return content.strip()

            if attempt < self._EMPTY_CONTENT_RETRIES:
                current_max_tokens = int(request["max_tokens"])
                request["max_tokens"] = min(
                    current_max_tokens * 2,
                    self._RETRY_MAX_TOKENS,
                )

        raise RuntimeError("DeepSeek response contained an empty answer")

    @staticmethod
    def _build_client() -> OpenAI:
        """Create the API client only when a generation request is made."""
        if settings is None:
            raise RuntimeError("Project settings are unavailable")

        api_key = str(settings.deepseek_api_key).strip()
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured")

        return OpenAI(
            api_key=api_key,
            base_url=settings.deepseek_base_url,
            timeout=settings.deepseek_timeout,
            max_retries=0,
        )
