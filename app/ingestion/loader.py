"""Load source documents and the configured production knowledge base."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from config.settings import settings

if TYPE_CHECKING:
    from app.ingestion.chunker import Chunk


@dataclass(frozen=True)
class Document:
    """A complete source document with its origin preserved."""

    content: str
    source: str


def canonical_source(file_path: str | Path, knowledge_base_path: str | Path) -> str:
    """Return a stable POSIX source path relative to the knowledge base root."""
    path = Path(file_path).resolve()
    root = Path(knowledge_base_path).resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(
            f"Source {path} is outside knowledge base {root}"
        ) from error


def load_txt(
    file_path: str | Path,
    *,
    source_root: str | Path | None = None,
) -> Document:
    """Load UTF-8 text and optionally store a canonical logical source path."""
    path = Path(file_path)
    return Document(
        content=path.read_text(encoding="utf-8"),
        source=(
            canonical_source(path, source_root)
            if source_root is not None
            else str(path)
        ),
    )


def configured_knowledge_base_path() -> Path:
    """Return the absolute path of the configured knowledge base directory."""
    if settings is None:
        raise RuntimeError("Project settings are unavailable")

    configured_path = Path(settings.knowledge_base_path)
    if not configured_path.is_absolute():
        configured_path = Path(__file__).parents[2] / configured_path
    return configured_path.resolve()


def load_configured_knowledge_base() -> list["Chunk"]:
    """Load and chunk the configured knowledge base deterministically."""
    if settings is None:
        raise RuntimeError("Project settings are unavailable")

    knowledge_base_path = configured_knowledge_base_path()
    if not knowledge_base_path.is_dir():
        raise FileNotFoundError(
            f"Configured knowledge base directory does not exist: {knowledge_base_path}"
        )

    # Keep loader and chunker as the single source of ingestion behavior.
    from app.ingestion.chunker import chunk_document

    chunks: list["Chunk"] = []
    for path in sorted(knowledge_base_path.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
            chunks.extend(
                chunk_document(
                    load_txt(path, source_root=knowledge_base_path),
                    chunk_size=settings.knowledge_base_chunk_size,
                )
            )
    return chunks
