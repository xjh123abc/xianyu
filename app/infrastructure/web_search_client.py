"""Small Tavily Search API client for technical model evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True, slots=True)
class WebSearchResult:
    """One normalized search result with traceable source data."""

    title: str
    url: str
    content: str
    score: float | None = None


class WebSearchError(RuntimeError):
    """Raised when a configured web search request cannot be completed."""


class TavilySearchClient:
    """Wrap Tavily Search without generating customer-facing answers."""

    endpoint = "https://api.tavily.com/search"

    def __init__(
        self,
        *,
        api_key: str,
        timeout_seconds: float,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key must not be blank")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        self.api_key = api_key.strip()
        self.timeout_seconds = timeout_seconds
        self._client = client

    def search(
        self,
        query: str,
        *,
        include_domains: Sequence[str],
        max_results: int = 3,
    ) -> list[WebSearchResult]:
        """Run one Tavily search and normalize the result list."""

        normalized_query = str(query or "").strip()
        if not normalized_query:
            raise ValueError("query must not be blank")
        if max_results <= 0:
            raise ValueError("max_results must be greater than zero")
        domains = [domain.strip() for domain in include_domains if domain.strip()]
        if not domains:
            raise ValueError("include_domains must not be empty")

        payload: dict[str, object] = {
            "query": normalized_query,
            "max_results": max_results,
            "include_answer": False,
            "include_domains": domains,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            if self._client is not None:
                response = self._client.post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
            else:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    response = client.post(
                        self.endpoint,
                        json=payload,
                        headers=headers,
                    )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise WebSearchError(f"Tavily search failed: {exc}") from exc
        return self._parse_results(data)

    @staticmethod
    def _parse_results(data: object) -> list[WebSearchResult]:
        if not isinstance(data, Mapping):
            raise WebSearchError("Tavily search returned an invalid payload")
        raw_results = data.get("results")
        if not isinstance(raw_results, list):
            raise WebSearchError("Tavily search returned no result list")

        results: list[WebSearchResult] = []
        for raw in raw_results:
            if not isinstance(raw, Mapping):
                continue
            title = _clean_text(raw.get("title"))
            url = _clean_text(raw.get("url"))
            content = _clean_text(raw.get("content"))
            if not title or not url or not content:
                continue
            score = _optional_float(raw.get("score"))
            results.append(
                WebSearchResult(title=title, url=url, content=content, score=score)
            )
        return results


def _clean_text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
