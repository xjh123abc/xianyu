from pathlib import Path

from app.ingestion.chunker import chunk_document
from app.ingestion.loader import load_txt


TEST_FILE = Path(__file__).parents[1] / "data" / "raw" / "ingestion_test.txt"


def test_load_txt_returns_complete_content_and_source() -> None:
    document = load_txt(TEST_FILE)

    assert document.content == TEST_FILE.read_text(encoding="utf-8")
    assert document.source == str(TEST_FILE)


def test_chunk_document_splits_content_and_preserves_source() -> None:
    document = load_txt(TEST_FILE)
    chunks = chunk_document(document, chunk_size=100)

    assert len(chunks) > 1
    assert all(chunk.content for chunk in chunks)
    assert all(chunk.source == document.source for chunk in chunks)
    assert all(len(chunk.content) <= 100 for chunk in chunks)
    assert "".join(chunk.content for chunk in chunks) == document.content
