"""Embedding service abstraction."""

from functools import lru_cache
from pathlib import Path
from typing import Sequence

from app.infrastructure.model_runtime import configure_huggingface_loading

configure_huggingface_loading()

from sentence_transformers import SentenceTransformer

from app.ingestion.chunker import Chunk
from config.settings import settings


@lru_cache(maxsize=1)
def _load_model(model_path: str) -> SentenceTransformer:
    """Load one local embedding model instance per process."""
    return SentenceTransformer(
        model_path,
        local_files_only=True,
    )


class EmbeddingService:
    """Create vectors for chunks with the configured local embedding model."""

    def __init__(self) -> None:
        if settings is None:
            raise RuntimeError("Project settings are unavailable")

        configured_path = str(settings.embedding_model_path).strip()
        if not configured_path:
            raise FileNotFoundError("Embedding model path is not configured")

        self.model_path = Path(configured_path)
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Embedding model path not found: {self.model_path}")

        self.model = _load_model(str(self.model_path))

    def embed_chunks(self, chunks: Sequence[Chunk]) -> list[list[float]]:
        """Generate one vector from each chunk's content."""
        if not chunks:
            return []

        contents = [chunk.content for chunk in chunks]
        vectors = self.model.encode(
            contents,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        if len(vectors) != len(chunks):
            raise RuntimeError("The number of vectors does not match the number of chunks")

        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Generate one vector for a user query with the same local model."""
        vector = self.model.encode(
            query,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vector.tolist()
