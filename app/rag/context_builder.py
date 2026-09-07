"""Build structured context for a future language-model call."""

from collections.abc import Mapping, Sequence
from typing import Any, TypedDict


class SourceReference(TypedDict):
    """Minimal source information kept outside the LLM context text."""

    source: str
    index: Any


class ContextBuildResult(TypedDict):
    """Structured separation of LLM text and program-level source data."""

    context: str
    sources: list[SourceReference]


class ContextBuilder:
    """Build context text without mixing in retrieval metadata."""

    def build_context(
        self,
        candidates: Sequence[Mapping[str, Any]] | None,
    ) -> ContextBuildResult:
        """Keep valid chunk text ordered and sources in a separate structure."""
        result: ContextBuildResult = {"context": "", "sources": []}
        if not candidates:
            return result

        contexts: list[str] = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue

            content = candidate.get("context")
            if content is None:
                content = candidate.get("content")
            if content is None or not str(content).strip():
                continue

            contexts.append(str(content))
            source = candidate.get("source")
            index = candidate.get("index")
            if index is None:
                index = candidate.get("chunk_index")
            result["sources"].append(
                {
                    "source": "" if source is None else str(source),
                    "index": index,
                }
            )

        result["context"] = "\n\n".join(contexts)
        return result

    def build(self, candidates: Sequence[Mapping[str, Any]] | None) -> ContextBuildResult:
        """Alias for build_context."""
        return self.build_context(candidates)
