"""Controlled web fallback for product model technical knowledge."""

from __future__ import annotations

import time
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse

from app.infrastructure.web_search_client import (
    TavilySearchClient,
    WebSearchError,
    WebSearchResult,
)
from config.settings import settings


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    fetched_at_monotonic: float
    prepared: dict[str, object]


class TechnicalKnowledgeService:
    """Fetch model-level technical evidence after local evidence is exhausted."""

    def __init__(
        self,
        *,
        search_client: TavilySearchClient | None = None,
        enabled: bool | None = None,
        allowed_domains: Sequence[str] | None = None,
        cache_ttl_seconds: int | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._enabled = (
            bool(getattr(settings, "tech_search_enabled", False))
            if enabled is None
            else bool(enabled)
        )
        self._allowed_domains = tuple(
            _split_domains(getattr(settings, "tech_search_allowed_domains", ""))
            if allowed_domains is None
            else [domain.strip().lower() for domain in allowed_domains if domain.strip()]
        )
        self._cache_ttl_seconds = int(
            getattr(settings, "tech_search_cache_ttl_seconds", 86400)
            if cache_ttl_seconds is None
            else cache_ttl_seconds
        )
        self._clock = clock or time.monotonic
        self._search_client = search_client
        self._cache: dict[tuple[str, str, tuple[str, ...]], _CacheEntry] = {}

    def prepare_model_evidence(
        self,
        question: str,
        *,
        item: Mapping[str, object],
    ) -> dict[str, object]:
        """Return web evidence for public model knowledge only."""

        model = self.confirm_model(item)
        if model is None:
            return _empty("product_model_unavailable")
        readiness_issue = self._readiness_issue()
        if readiness_issue is not None:
            return _empty(readiness_issue)

        cache_key = (model.casefold(), question.strip().casefold(), self._allowed_domains)
        cached = self._cache.get(cache_key)
        now = self._clock()
        if cached is not None and now - cached.fetched_at_monotonic <= self._cache_ttl_seconds:
            return dict(cached.prepared)

        query = self._search_query(question, model)
        try:
            results = self._client().search(
                query,
                include_domains=self._allowed_domains,
                max_results=3,
            )
        except (WebSearchError, ValueError, TimeoutError, OSError):
            return _empty("technical_search_failed")

        accepted = [
            result
            for result in results
            if self._source_allowed(result.url)
            and self._matches_model(result, model)
        ]
        if not accepted:
            return _empty("technical_search_evidence_unavailable")

        fetched_at = datetime.now(timezone.utc).isoformat()
        contexts = [
            f"[web] {result.title}\nURL: {result.url}\nFetched at: {fetched_at}\n{result.content}"
            for result in accepted
        ]
        sources = [
            {
                "source": result.url,
                "index": fetched_at,
                "title": result.title,
                "fetched_at": fetched_at,
                "source_kind": "web_model_knowledge",
                "product_model": model,
            }
            for result in accepted
        ]
        prepared: dict[str, object] = {
            "can_answer": True,
            "context": {"context": "\n\n".join(contexts), "sources": sources},
            "sources": sources,
            "results": [
                {
                    "title": result.title,
                    "url": result.url,
                    "content": result.content,
                    "score": result.score,
                    "fetched_at": fetched_at,
                    "source_kind": "web_model_knowledge",
                    "product_model": model,
                }
                for result in accepted
            ],
            "reliability": None,
            "issues": [],
            "knowledge_scope": "product",
            "product_model": model,
            "source_kind": "web_model_knowledge",
        }
        self._cache[cache_key] = _CacheEntry(now, dict(prepared))
        return prepared

    @staticmethod
    def confirm_model(item: Mapping[str, object]) -> str | None:
        """Read a seller-confirmed public model identifier from item facts."""

        facts = item.get("facts")
        identity = facts.get("identity") if isinstance(facts, Mapping) else None
        brand = _known_text(identity.get("brand")) if isinstance(identity, Mapping) else None
        model = _known_text(identity.get("model")) if isinstance(identity, Mapping) else None
        if model is None:
            return None
        return " ".join(part for part in (brand, model) if part)

    def _readiness_issue(self) -> str | None:
        if not self._enabled:
            return "technical_search_not_enabled"
        if not self._allowed_domains:
            return "technical_search_allowed_domains_unconfigured"
        if self._search_client is None and not str(getattr(settings, "tavily_api_key", "")).strip():
            return "technical_search_api_key_unconfigured"
        if self._cache_ttl_seconds <= 0:
            return "technical_search_cache_ttl_invalid"
        return None

    def _client(self) -> TavilySearchClient:
        if self._search_client is None:
            self._search_client = TavilySearchClient(
                api_key=str(getattr(settings, "tavily_api_key", "")),
                timeout_seconds=float(
                    getattr(settings, "tech_search_timeout_seconds", 6.0)
                ),
            )
        return self._search_client

    @staticmethod
    def _search_query(question: str, model: str) -> str:
        return f"{model} {question}".strip()

    def _source_allowed(self, url: str) -> bool:
        host = urlparse(url).hostname
        if host is None:
            return False
        normalized = host.lower().removeprefix("www.")
        for domain in self._allowed_domains:
            allowed = domain.lower().removeprefix("www.")
            if normalized == allowed or normalized.endswith("." + allowed):
                return True
        return False

    @staticmethod
    def _matches_model(result: WebSearchResult, model: str) -> bool:
        haystack = f"{result.title}\n{result.content}".casefold()
        tokens = [token.casefold() for token in re.findall(r"[A-Za-z0-9]+", model)]
        if not tokens:
            return model.casefold() in haystack
        return all(token in haystack for token in tokens)


def _empty(issue: str) -> dict[str, object]:
    return {
        "can_answer": False,
        "context": None,
        "sources": [],
        "results": [],
        "reliability": None,
        "issues": [issue],
    }


def _known_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized or normalized == "unknown":
        return None
    return normalized


def _split_domains(value: object) -> list[str]:
    if not isinstance(value, str):
        return []
    return [part.strip() for part in value.split(",") if part.strip()]
