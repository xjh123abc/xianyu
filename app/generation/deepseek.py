"""DeepSeek answer generation through the OpenAI-compatible API."""

from typing import Any

from openai import OpenAI

from app.generation.prompt import build_messages
from config.settings import settings


class DeepSeekGenerator:
    """Generate a grounded answer from a query and ContextBuilder text."""

    def __init__(self, client: Any | None = None) -> None:
        """Accept an injected client for tests; otherwise build it lazily."""
        self.client = client

    def generate(self, query: str, context: str) -> str:
        """Send query and context to DeepSeek and return its answer text."""
        if settings is None:
            raise RuntimeError("Project settings are unavailable")

        messages = build_messages(query, context)
        client = self.client or self._build_client()
        response = client.chat.completions.create(
            model=settings.deepseek_model,
            messages=messages,
            temperature=settings.deepseek_temperature,
            max_tokens=settings.deepseek_max_tokens,
            stream=False,
        )

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as error:
            raise RuntimeError("DeepSeek response did not contain an answer") from error

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("DeepSeek response contained an empty answer")
        return content.strip()

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
        )
