"""Split loaded documents into retrieval chunks."""

from dataclasses import dataclass

from app.ingestion.loader import Document


@dataclass(frozen=True)
class Chunk:
    """A deterministic text chunk with its original source preserved."""

    content: str
    source: str


def chunk_document(document: Document, chunk_size: int = 500) -> list[Chunk]:
    """Split a document into fixed-length chunks without semantic processing."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    return [
        Chunk(
            content=document.content[start : start + chunk_size],
            source=document.source,
        )
        for start in range(0, len(document.content), chunk_size)
    ]
