"""MediaWiki search retrieval."""

from __future__ import annotations

from typing import Any

import requests

from .base import LRUCache, RetryingHTTPProvider, WebSearchProvider


class MediaWikiSearchProvider(RetryingHTTPProvider, WebSearchProvider):
    """Read document and corpus counts from a MediaWiki Action API."""

    def __init__(
        self,
        *,
        lang: str = "en",
        project: str = "wikipedia",
        session: requests.Session | None = None,
        cache_size: int = 2000,
        timeout: float = 10.0,
        max_attempts: int = 3,
        backoff: float = 0.5,
        max_backoff: float = 6.0,
        fallback_corpus_size: int = 10**8,
        user_agent: str = "agentoi/0.1",
    ) -> None:
        self.lang = lang
        self.project = project
        self.base_url = f"https://{lang}.{project}.org/w/api.php"
        http_session = session or requests.Session()
        http_session.headers.update({"User-Agent": user_agent})
        self._configure_http(
            session=http_session,
            timeout=timeout,
            max_attempts=max_attempts,
            backoff=backoff,
            max_backoff=max_backoff,
        )
        self._counts = LRUCache(cache_size)
        self._corpus_size_cache: int | None = None
        self.fallback_corpus_size = max(1, int(fallback_corpus_size))

    @staticmethod
    def _quote_phrase(term: str) -> str:
        term = (term or "").strip()
        if " " in term and not (term.startswith('"') and term.endswith('"')):
            return f'"{term}"'
        return term

    def _search(self, query: str) -> int:
        cached = self._counts.get_cached(query)
        if cached is not None:
            return cached
        params: dict[str, Any] = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srwhat": "text",
            "srinfo": "totalhits",
            "srlimit": 1,
            "format": "json",
            "utf8": 1,
        }
        data = self._request_json(self.base_url, params)
        try:
            count = max(
                0,
                int((data or {}).get("query", {}).get("searchinfo", {}).get("totalhits", 0)),
            )
        except (TypeError, ValueError):
            count = 0
        if data is not None:
            self._counts.put_cached(query, count)
        return count

    def search_count(self, query: str) -> int:
        return self._search(self._quote_phrase(query))

    def co_occurrence_count(self, term1: str, term2: str) -> int:
        return self._search(
            f"{self._quote_phrase(term1)} {self._quote_phrase(term2)}".strip()
        )

    def corpus_size(self) -> int:
        if self._corpus_size_cache is not None:
            return self._corpus_size_cache
        data = self._request_json(
            self.base_url,
            {
                "action": "query",
                "meta": "siteinfo",
                "siprop": "statistics",
                "format": "json",
                "utf8": 1,
            },
        )
        stats = (data or {}).get("query", {}).get("statistics", {})
        try:
            size = int(stats.get("articles") or stats.get("pages") or 0)
        except (AttributeError, TypeError, ValueError):
            size = 0
        self._corpus_size_cache = size if size > 0 else self.fallback_corpus_size
        return self._corpus_size_cache
