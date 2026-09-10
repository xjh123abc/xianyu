"""Platform-specific safeguards for local Hugging Face model loading."""

from __future__ import annotations

import os
import sys


def configure_huggingface_loading() -> None:
    """Avoid unstable asynchronous weight materialization on Windows."""

    if sys.platform == "win32":
        os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
