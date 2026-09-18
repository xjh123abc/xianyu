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


def test_auto_model_device_prefers_cuda_when_available(monkeypatch) -> None:
    monkeypatch.setattr(model_runtime, "_cuda_available", lambda: True)

    assert model_runtime.resolve_model_device() == "cuda"


def test_auto_model_device_uses_cpu_when_cuda_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(model_runtime, "_cuda_available", lambda: False)

    assert model_runtime.resolve_model_device("auto") == "cpu"


def test_explicit_cuda_without_a_gpu_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(model_runtime, "_cuda_available", lambda: False)

    try:
        model_runtime.resolve_model_device("cuda")
    except RuntimeError as error:
        assert "CUDA is unavailable" in str(error)
    else:
        raise AssertionError("explicit CUDA must fail without CUDA support")


def test_auto_device_retries_the_loader_on_cpu_after_a_cuda_failure(monkeypatch) -> None:
    monkeypatch.setattr(model_runtime, "_cuda_available", lambda: True)
    calls: list[str] = []

    def loader(device: str) -> str:
        calls.append(device)
        if device == "cuda":
            raise RuntimeError("CUDA out of memory")
        return "cpu-model"

    assert model_runtime.load_with_device_fallback(loader) == "cpu-model"
    assert calls == ["cuda", "cpu"]


def test_auto_device_does_not_hide_a_non_cuda_loader_failure(monkeypatch) -> None:
    monkeypatch.setattr(model_runtime, "_cuda_available", lambda: True)
    calls: list[str] = []

    def loader(device: str) -> str:
        calls.append(device)
        raise RuntimeError("model configuration is invalid")

    try:
        model_runtime.load_with_device_fallback(loader)
    except RuntimeError as error:
        assert str(error) == "model configuration is invalid"
    else:
        raise AssertionError("non-CUDA model failures must remain visible")
    assert calls == ["cuda"]
