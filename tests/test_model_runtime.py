"""Tests for platform-specific local model loading safeguards."""

from app.infrastructure import model_runtime


def test_windows_disables_async_huggingface_loading(monkeypatch) -> None:
    monkeypatch.delenv("HF_DEACTIVATE_ASYNC_LOAD", raising=False)
    monkeypatch.setattr(model_runtime.sys, "platform", "win32")

    model_runtime.configure_huggingface_loading()

    assert model_runtime.os.environ["HF_DEACTIVATE_ASYNC_LOAD"] == "1"


def test_explicit_huggingface_loading_choice_is_preserved(monkeypatch) -> None:
    monkeypatch.setenv("HF_DEACTIVATE_ASYNC_LOAD", "0")
    monkeypatch.setattr(model_runtime.sys, "platform", "win32")

    model_runtime.configure_huggingface_loading()

    assert model_runtime.os.environ["HF_DEACTIVATE_ASYNC_LOAD"] == "0"
