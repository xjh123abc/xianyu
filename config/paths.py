"""Stable filesystem-path helpers shared by API and command-line entry points."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(value: str | Path, *, project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve a relative path from the agreed project root.

    Absolute paths are deliberately left rooted where the caller supplied
    them.  This keeps an explicit command-line database override explicit.
    """

    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (Path(project_root).resolve() / candidate).resolve()
