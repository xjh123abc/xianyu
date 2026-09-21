"""Platform-specific safeguards for local Hugging Face model loading."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import TypeVar


Model = TypeVar("Model")


def configure_huggingface_loading() -> None:
    """Avoid unstable asynchronous weight materialization on Windows."""

    if sys.platform == "win32":
        os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")


def _cuda_available() -> bool:
    """Defer the heavyweight Torch import until a model is actually loaded."""

    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def resolve_model_device(requested: str = "auto") -> str:
    """Choose the configured local-model device without silently ignoring it."""

    normalized = str(requested or "auto").strip().lower()
    if normalized not in {"auto", "cuda", "cpu"}:
        raise ValueError("MODEL_DEVICE must be one of: auto, cuda, cpu")
    if normalized == "cpu":
        return "cpu"
    if _cuda_available():
        return "cuda"
    if normalized == "cuda":
        raise RuntimeError("MODEL_DEVICE=cuda was requested but CUDA is unavailable")
    return "cpu"


def _is_cuda_load_failure(error: OSError | RuntimeError) -> bool:
    """Avoid treating invalid model files or code defects as a GPU fallback."""

    text = str(error).casefold()
    return any(
        token in text
        for token in ("cuda", "cudnn", "cublas", "cufft", "gpu", "out of memory")
    )


def load_with_device_fallback(
    loader: Callable[[str], Model],
    requested: str = "auto",
) -> Model:
    """Prefer CUDA for ``auto`` but retain a working CPU service on GPU failure."""

    normalized = str(requested or "auto").strip().lower()
    device = resolve_model_device(normalized)
    try:
        return loader(device)
    except (OSError, RuntimeError) as error:
        if (
            normalized != "auto"
            or device != "cuda"
            or not _is_cuda_load_failure(error)
        ):
            raise
        try:
            import torch

            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass
        return loader("cpu")
