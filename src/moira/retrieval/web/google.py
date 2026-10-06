"""Google Programmable Search retrieval."""

from __future__ import annotations

import os

import requests

from .base import LRUCache, RetryingHTTPProvider, WebSearchProvider


class GoogleCSEProvider(RetryingHTTPProvider, WebSearchProvider):
    """Read document counts from Google's Custom Search JSON API."""

    BASE_URL = "https://www.googleapis.com/customsearch/v1"

    def __init__(
        self,
        api_key: str | None = None,
        cx: str | None = None,
        *,
        session: requests.Session | None = None,
        cache_size: int = 2000,
        timeout: float = 10.0,
        max_attempts: int = 3,
        backoff: float = 0.5,
        max_backoff: float = 8.0,
        corpus_size: int = 10**12,
        user_agent: str = "moira/0.1",
    ) -> None:
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.cx = cx or os.getenv("GOOGLE_CSE_CX")
        if not self.api_key or not self.cx:
            raise ValueError(
                "Google CSE requires GOOGLE_API_KEY and GOOGLE_CSE_CX"
            )
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
        self._corpus_size = max(1, int(corpus_size))

    @staticmethod
    def _normalize(query: str) -> str:
        query = (query or "").strip()
        identifier = query.startswith(("http://", "https://")) or any(
            token in query for token in (":", "/")
        )
        if (
            " " in query
            and not identifier
            and not (query.startswith('"') and query.endswith('"'))
        ):
            return f'"{query}"'
        return query

    def _count(self, query: str) -> int:
        cached = self._counts.get_cached(query)
        if cached is not None:
            return cached
        data = self._request_json(
            self.BASE_URL,
            {
                "key": self.api_key,
                "cx": self.cx,
                "q": query,
                "num": 1,
                "safe": "off",
                "fields": "searchInformation(totalResults)",
            },
        )
        try:
            count = max(
                0,
                int((data or {}).get("searchInformation", {}).get("totalResults", 0)),
            )
        except (TypeError, ValueError):
            count = 0
        if data is not None:
            self._counts.put_cached(query, count)
        return count

    def search_count(self, query: str) -> int:
        return self._count(self._normalize(query))

    def co_occurrence_count(self, term1: str, term2: str) -> int:
        return self._count(
            f"{self._normalize(term1)} {self._normalize(term2)}".strip()
        )

    def corpus_size(self) -> int:
        return self._corpus_size
