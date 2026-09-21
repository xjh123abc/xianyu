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
    scope: str | None = None
    item_id: str | None = None


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
    scope: str | None = None,
    item_id: str | None = None,
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
        scope=scope,
        item_id=item_id,
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
    return load_knowledge_base(knowledge_base_path)


def load_knowledge_base(
    knowledge_base_path: str | Path,
    *,
    scoped: bool = False,
) -> list["Chunk"]:
    """Load and chunk a specific corpus, optionally deriving Xianyu scopes."""

    knowledge_base_path = Path(knowledge_base_path).resolve()
    if not knowledge_base_path.is_dir():
        raise FileNotFoundError(
            f"Configured knowledge base directory does not exist: {knowledge_base_path}"
        )

    # Keep loader and chunker as the single source of ingestion behavior.
    from app.ingestion.chunker import chunk_document

    chunks: list["Chunk"] = []
    paths = sorted(
        (path for path in knowledge_base_path.rglob("*") if path.is_file()),
        key=lambda item: item.as_posix(),
    )
    for path in paths:
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        scope = None
        item_id = None
        source = canonical_source(path, knowledge_base_path)
        if scoped:
            parts = Path(source).parts
            if len(parts) >= 2 and parts[0] == "common":
                scope = "common"
            elif len(parts) == 2 and parts[0] == "items":
                scope = "item"
                item_id = Path(parts[1]).stem
            if scope is None or (scope == "item" and not item_id):
                raise ValueError(
                    "Scoped Xianyu knowledge files must be under common/ or items/<item_id>.md"
                )
        chunks.extend(
            chunk_document(
                load_txt(
                    path,
                    source_root=knowledge_base_path,
                    scope=scope,
                    item_id=item_id,
                ),
                chunk_size=settings.knowledge_base_chunk_size,
            )
        )
    return chunks


def configured_xianyu_knowledge_base_path() -> Path:
    """Resolve the configured Xianyu knowledge corpus."""

    if settings is None:
        raise RuntimeError("Project settings are unavailable")
    path = Path(settings.xianyu_knowledge_base_path)
    if not path.is_absolute():
        path = Path(__file__).parents[2] / path
    return path.resolve()


def load_xianyu_knowledge_base() -> list["Chunk"]:
    """Load only the seller-confirmed Xianyu corpus with scope metadata."""

    return load_knowledge_base(configured_xianyu_knowledge_base_path(), scoped=True)
