"""Load source documents from the raw data directory."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    """A complete source document with its origin preserved."""

    content: str
    source: str


def load_txt(file_path: str | Path) -> Document:
    """Load a UTF-8 TXT file and return its complete content and source path."""
    path = Path(file_path)
    return Document(
        content=path.read_text(encoding="utf-8"),
        source=str(path),
    )
