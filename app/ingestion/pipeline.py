"""Repeatable local knowledge-base ingestion and version checking."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence, TypedDict

from app.ingestion.chunker import Chunk
from app.ingestion.loader import (
    configured_knowledge_base_path,
    configured_xianyu_knowledge_base_path,
    load_configured_knowledge_base,
    load_xianyu_knowledge_base,
)
from app.infrastructure.qdrant import QdrantStore
from app.services.embedding_service import EmbeddingService
from config.settings import settings


class IngestionResult(TypedDict):
    """Summary written to the successful-ingestion manifest."""

    fingerprint: str
    chunk_count: int
    knowledge_base_path: str


def knowledge_base_fingerprint(chunks: Sequence[Chunk]) -> str:
    """Return a deterministic fingerprint for the exact retrieval corpus."""
    digest = hashlib.sha256()
    for chunk_index, chunk in enumerate(chunks):
        record = {
            "content": chunk.content,
            "source": chunk.source,
            "chunk_index": chunk_index,
            "scope": chunk.scope,
            "item_id": chunk.item_id,
        }
        digest.update(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            .encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def configured_manifest_path() -> Path:
    """Resolve the local manifest path from project settings."""
    if settings is None:
        raise RuntimeError("Project settings are unavailable")

    path = Path(settings.ingestion_manifest_path)
    if not path.is_absolute():
        path = Path(__file__).parents[2] / path
    return path.resolve()


def _write_manifest(path: Path, result: IngestionResult) -> None:
    """Atomically write a manifest only after Qdrant ingestion succeeds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def ensure_knowledge_base_in_sync(
    chunks: Sequence[Chunk] | None = None,
    *,
    manifest_path: str | Path | None = None,
) -> IngestionResult:
    """Reject online retrieval when local chunks differ from last successful ingest."""
    current_chunks = list(chunks) if chunks is not None else load_configured_knowledge_base()
    path = Path(manifest_path) if manifest_path is not None else configured_manifest_path()
    if not path.is_file():
        raise RuntimeError(
            "Knowledge base is not synchronized with Qdrant: ingestion manifest is missing. "
            "Run `python -m app.ingestion.pipeline`."
        )

    try:
        manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid knowledge-base ingestion manifest: {path}") from error

    fingerprint = knowledge_base_fingerprint(current_chunks)
    if manifest.get("fingerprint") != fingerprint:
        raise RuntimeError(
            "Knowledge base is not synchronized with Qdrant: local chunks changed since "
            "the last successful ingestion. Run `python -m app.ingestion.pipeline`."
        )
    return manifest  # type: ignore[return-value]


class IngestionPipeline:
    """Load, chunk, embed, and replace the configured Qdrant corpus."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        store: QdrantStore | None = None,
        *,
        manifest_path: str | Path | None = None,
        knowledge_base_path: str | Path | None = None,
        xianyu: bool = False,
    ) -> None:
        if xianyu and knowledge_base_path is not None:
            raise ValueError("Pass either xianyu=True or knowledge_base_path, not both")
        self.xianyu = xianyu
        self.knowledge_base_path = (
            Path(knowledge_base_path).resolve()
            if knowledge_base_path is not None
            else configured_xianyu_knowledge_base_path() if xianyu else None
        )
        self.embedding_service = embedding_service or EmbeddingService()
        self.store = store or QdrantStore(
            collection_name=settings.xianyu_qdrant_collection if xianyu else None,
            knowledge_base_path=self.knowledge_base_path,
            corpus_id=(
                settings.xianyu_corpus_id if xianyu else settings.knowledge_corpus_id
            ),
        )
        self.manifest_path = (
            Path(manifest_path) if manifest_path is not None else configured_manifest_path()
        )

    def ingest(self) -> IngestionResult:
        """Ingest the current configured corpus and then record its fingerprint."""
        chunks = load_xianyu_knowledge_base() if self.xianyu else load_configured_knowledge_base()
        fingerprint = knowledge_base_fingerprint(chunks)
        vectors = self.embedding_service.embed_chunks(chunks)
        self.store.upsert_chunks(chunks, vectors)

        result: IngestionResult = {
            "fingerprint": fingerprint,
            "chunk_count": len(chunks),
            "knowledge_base_path": str(
                self.knowledge_base_path or configured_knowledge_base_path()
            ),
        }
        _write_manifest(self.manifest_path, result)
        return result


def main() -> None:
    """Run the local ingestion command."""
    parser = argparse.ArgumentParser(description="Ingest the configured RAG knowledge base")
    parser.add_argument("--manifest", type=Path, help="Optional manifest output path")
    parser.add_argument(
        "--scenario",
        choices=("ecommerce", "xianyu"),
        default="ecommerce",
        help="Corpus to ingest (default: ecommerce)",
    )
    args = parser.parse_args()
    result = IngestionPipeline(
        manifest_path=args.manifest
        or (settings.xianyu_ingestion_manifest_path if args.scenario == "xianyu" else None),
        xianyu=args.scenario == "xianyu",
    ).ingest()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
