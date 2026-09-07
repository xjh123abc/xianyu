import json
import shutil
from pathlib import Path

import pytest

from app.ingestion.chunker import Chunk
from app.ingestion import pipeline as ingestion_module
from app.ingestion.pipeline import (
    IngestionPipeline,
    ensure_knowledge_base_in_sync,
    knowledge_base_fingerprint,
)


@pytest.fixture
def local_test_dir():
    path = Path(__file__).parent / ".test-ingestion-artifacts"
    if path.exists():
        shutil.rmtree(path)
    path.mkdir()
    try:
        yield path
    finally:
        if path.exists():
            shutil.rmtree(path)


def test_fingerprint_is_stable_and_changes_with_content() -> None:
    chunks = [Chunk(content="same", source="a.md")]

    assert knowledge_base_fingerprint(chunks) == knowledge_base_fingerprint(list(chunks))
    assert knowledge_base_fingerprint(chunks) != knowledge_base_fingerprint(
        [Chunk(content="changed", source="a.md")]
    )


class FakeEmbeddingService:
    def __init__(self) -> None:
        self.received = None

    def embed_chunks(self, chunks):
        self.received = list(chunks)
        return [[1.0, 2.0] for _ in chunks]


class FakeStore:
    def __init__(self, error: Exception | None = None) -> None:
        self.received = None
        self.error = error

    def upsert_chunks(self, chunks, vectors) -> None:
        if self.error:
            raise self.error
        self.received = (list(chunks), list(vectors))


def test_ingestion_reuses_configured_loader_and_writes_manifest_after_qdrant(
    monkeypatch: pytest.MonkeyPatch,
    local_test_dir: Path,
) -> None:
    chunks = [Chunk(content="shipping", source="shipping.md")]
    embedding = FakeEmbeddingService()
    store = FakeStore()
    monkeypatch.setattr(ingestion_module, "load_configured_knowledge_base", lambda: chunks)
    monkeypatch.setattr(
        ingestion_module,
        "configured_knowledge_base_path",
        lambda: local_test_dir / "knowledge_base",
    )

    manifest_path = local_test_dir / "manifest.json"
    result = IngestionPipeline(
        embedding_service=embedding,
        store=store,
        manifest_path=manifest_path,
    ).ingest()

    assert embedding.received == chunks
    assert store.received == (chunks, [[1.0, 2.0]])
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == result
    assert result["fingerprint"] == knowledge_base_fingerprint(chunks)


def test_ingestion_failure_does_not_write_success_manifest(
    monkeypatch: pytest.MonkeyPatch,
    local_test_dir: Path,
) -> None:
    chunks = [Chunk(content="shipping", source="shipping.md")]
    monkeypatch.setattr(ingestion_module, "load_configured_knowledge_base", lambda: chunks)
    manifest_path = local_test_dir / "manifest.json"

    with pytest.raises(RuntimeError, match="qdrant failed"):
        IngestionPipeline(
            embedding_service=FakeEmbeddingService(),
            store=FakeStore(RuntimeError("qdrant failed")),
            manifest_path=manifest_path,
        ).ingest()

    assert not manifest_path.exists()


def test_sync_check_accepts_matching_manifest_and_rejects_changed_chunks(
    local_test_dir: Path,
) -> None:
    chunks = [Chunk(content="shipping", source="shipping.md")]
    manifest_path = local_test_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "fingerprint": knowledge_base_fingerprint(chunks),
                "chunk_count": 1,
                "knowledge_base_path": str(local_test_dir),
            }
        ),
        encoding="utf-8",
    )

    assert ensure_knowledge_base_in_sync(chunks, manifest_path=manifest_path)["chunk_count"] == 1
    with pytest.raises(RuntimeError, match="local chunks changed"):
        ensure_knowledge_base_in_sync(
            [Chunk(content="changed", source="shipping.md")],
            manifest_path=manifest_path,
        )


def test_sync_check_rejects_missing_manifest(local_test_dir: Path) -> None:
    with pytest.raises(RuntimeError, match="manifest is missing"):
        ensure_knowledge_base_in_sync([], manifest_path=local_test_dir / "missing.json")
