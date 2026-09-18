"""Verify that the configured local embedding and reranker models can infer.

The script loads both models into one process and runs a tiny inference through
each.  It is intentionally read-only: it does not contact Qdrant, call an LLM,
or modify model files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

# ``python scripts/verify_local_models.py`` puts ``scripts/`` rather than the
# repository root on ``sys.path``.  Keep this diagnostic usable in the same
# direct-execution form as the project's other scripts.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return requested


def _path(setting_name: str, value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"{setting_name} is not a directory: {path}")
    return str(path)


def _gpu_memory() -> dict[str, float] | None:
    if not torch.cuda.is_available():
        return None
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    arguments = parser.parse_args()

    if settings is None:
        raise RuntimeError("Project settings are unavailable")
    device = _resolve_device(arguments.device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    embedding_path = _path("EMBEDDING_MODEL_PATH", settings.embedding_model_path)
    reranker_path = _path("RERANKER_MODEL_PATH", settings.reranker_model_path)

    embedding = SentenceTransformer(
        embedding_path,
        device=device,
        local_files_only=True,
    )
    vector = embedding.encode("camera shipping policy", convert_to_numpy=True)

    reranker = CrossEncoder(
        reranker_path,
        device=device,
        local_files_only=True,
    )
    score = reranker.predict([("How fast do you ship?", "Orders ship within 48 hours.")])

    if device == "cuda":
        torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "requested_device": arguments.device,
                "resolved_device": device,
                "embedding_dimensions": int(len(vector)),
                "reranker_score_count": int(len(score)),
                "gpu_memory": _gpu_memory(),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
