from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.generation import deepseek as deepseek_module
from app.generation.deepseek import DeepSeekGenerator
from app.generation.prompt import build_messages
from config.settings import Settings


def test_build_messages_contains_query_and_context() -> None:
    messages = build_messages("商品破损怎么办？", "请保留包装并拍照。")

    assert messages[0]["role"] == "system"
    assert messages[1] == {
        "role": "user",
        "content": "用户问题：\n商品破损怎么办？\n\n参考资料：\n请保留包装并拍照。",
    }


def test_generator_sends_context_to_deepseek_and_returns_answer() -> None:
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="请保留包装并提交照片。")
            )
        ]
    )
    generator = DeepSeekGenerator(client=client)

    answer = generator.generate("商品破损怎么办？", "请保留包装并拍照。")

    assert answer == "请保留包装并提交照片。"
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == deepseek_module.settings.deepseek_model
    assert kwargs["messages"][1]["content"] == (
        "用户问题：\n商品破损怎么办？\n\n参考资料：\n请保留包装并拍照。"
    )
    assert kwargs["temperature"] == deepseek_module.settings.deepseek_temperature
    assert kwargs["max_tokens"] == deepseek_module.settings.deepseek_max_tokens
    assert kwargs["stream"] is False
    if deepseek_module.settings.deepseek_model == "deepseek-flash":
        assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


def test_generator_requires_api_key_only_when_building_real_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(deepseek_module.settings, "deepseek_api_key", "")

    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        DeepSeekGenerator().generate("问题", "证据")


def test_generator_builds_real_client_from_configured_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_openai(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(deepseek_module.settings, "deepseek_api_key", "test-key")
    monkeypatch.setattr(deepseek_module, "OpenAI", fake_openai)

    DeepSeekGenerator()._build_client()

    assert calls == [
        {
            "api_key": "test-key",
            "base_url": deepseek_module.settings.deepseek_base_url,
            "timeout": deepseek_module.settings.deepseek_timeout,
            "max_retries": 0,
        }
    ]


def test_settings_reads_deepseek_key_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment-key")

    assert Settings().deepseek_api_key == "environment-key"


def test_generator_rejects_empty_deepseek_answer() -> None:
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="   "))]
    )

    with pytest.raises(RuntimeError, match="empty answer"):
        DeepSeekGenerator(client=client).generate("问题", "证据")

    assert client.chat.completions.create.call_count == 2


def test_generator_retries_empty_answer_with_more_output_tokens() -> None:
    client = Mock()
    client.chat.completions.create.side_effect = [
        SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="   "))]
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="有依据的回答"))]
        ),
    ]

    answer = DeepSeekGenerator(client=client).generate("问题", "证据")

    assert answer == "有依据的回答"
    first_request = client.chat.completions.create.call_args_list[0].kwargs
    retry_request = client.chat.completions.create.call_args_list[1].kwargs
    assert first_request["max_tokens"] == deepseek_module.settings.deepseek_max_tokens
    assert retry_request["max_tokens"] == min(
        deepseek_module.settings.deepseek_max_tokens * 2,
        DeepSeekGenerator._RETRY_MAX_TOKENS,
    )


@pytest.mark.parametrize(
    "query, context",
    [("", "证据"), ("问题", "   ")],
)
def test_prompt_rejects_empty_query_or_context(query: str, context: str) -> None:
    with pytest.raises(ValueError):
        build_messages(query, context)
