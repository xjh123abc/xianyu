"""Result reranking."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from app.infrastructure.model_runtime import (
    configure_huggingface_loading,
    load_with_device_fallback,
)

configure_huggingface_loading()

from sentence_transformers import CrossEncoder

from config.settings import settings


class Reranker:
    """Rerank retrieved chunks with a locally loaded CrossEncoder model."""

    def __init__(
        self,
        model: CrossEncoder | None = None,
        *,
        scorer: Any | None = None,
    ) -> None:
        if model is not None and scorer is not None:
            raise ValueError("Pass either model or scorer, not both")

        self.model = model if model is not None else scorer
        if self.model is None:
            self.model = self._load_model()

    def rerank(
        self,
        query: str,
        candidates: Sequence[Mapping[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Score candidates in one batch and return the highest-scoring chunks."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not candidates:
            return []

        valid_candidates = [
            candidate
            for candidate in candidates
            if isinstance(candidate.get("content"), str)
            and candidate["content"].strip()
        ]
        if not valid_candidates:
            return []

        pairs = [(query, candidate["content"]) for candidate in valid_candidates]
        scores = self._predict(pairs)
        if len(scores) != len(valid_candidates):
            raise RuntimeError("The number of rerank scores does not match candidates")

        reranked = []
        for candidate, score in zip(valid_candidates, scores):
            result = dict(candidate)
            result["rerank_score"] = float(score)
            reranked.append(result)

        reranked.sort(key=lambda result: -result["rerank_score"])
        return reranked[:top_k]

    def _predict(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """Run one batched CrossEncoder prediction."""
        scores = self.model.predict(pairs)
        try:
            return [float(score) for score in scores]
        except TypeError:
            return [float(scores)]

    @staticmethod
    def _load_model() -> CrossEncoder:
        """Load the configured reranker strictly from a local model directory."""
        if settings is None:
            raise RuntimeError("Project settings are unavailable")

        configured_path = str(getattr(settings, "reranker_model_path", "")).strip()
        if not configured_path:
            raise FileNotFoundError("Reranker model path is not configured")

        model_path = Path(configured_path)
        if not model_path.is_dir():
            raise FileNotFoundError(f"Reranker model path not found: {model_path}")

        return load_with_device_fallback(
            lambda device: CrossEncoder(
                str(model_path),
                device=device,
                local_files_only=True,
            ),
            settings.model_device,
        )
