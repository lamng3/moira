"""Interfaces and HTTP utilities for web-search retrieval."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)


class LRUCache(OrderedDict):
    """Small in-memory least-recently-used cache."""

    def __init__(self, capacity: int = 1000):
        super().__init__()
        self.capacity = max(1, int(capacity))

    def get_cached(self, key: Any, default: Any = None) -> Any:
        if key in self:
            self.move_to_end(key)
            return self[key]
        return default

    def put_cached(self, key: Any, value: Any) -> None:
        self[key] = value
        self.move_to_end(key)
        if len(self) > self.capacity:
            self.popitem(last=False)


class WebSearchProvider(ABC):
    """Supply document counts for web-based similarity."""

    @abstractmethod
    def search_count(self, query: str) -> int:
        """Return the number of documents matching ``query``."""

    @abstractmethod
    def co_occurrence_count(self, term1: str, term2: str) -> int:
        """Return the number of documents matching both terms."""

    def corpus_size(self) -> int:
        """Return the searchable corpus size."""
        return 10**12


class RetryingHTTPProvider:
    """Shared bounded JSON request behavior."""

    _RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

    def _configure_http(
        self,
        *,
        session: requests.Session,
        timeout: float,
        max_attempts: int,
        backoff: float,
        max_backoff: float,
    ) -> None:
        self.session = session
        self.timeout = max(0.0, float(timeout))
        self.max_attempts = max(1, int(max_attempts))
        self.backoff = max(0.0, float(backoff))
        self.max_backoff = max(self.backoff, float(max_backoff))

    def _request_json(
        self, url: str, params: dict[str, Any]
    ) -> dict[str, Any] | None:
        delay = self.backoff
        for attempt in range(self.max_attempts):
            try:
                response = self.session.get(
                    url, params=params, timeout=self.timeout
                )
                if response.status_code in self._RETRYABLE_STATUS_CODES:
                    if attempt + 1 < self.max_attempts and delay:
                        time.sleep(delay)
                    delay = min(self.max_backoff, delay * 2)
                    continue
                response.raise_for_status()
                data = response.json()
                return data if isinstance(data, dict) else {}
            except (requests.RequestException, ValueError) as error:
                if attempt + 1 < self.max_attempts:
                    if delay:
                        time.sleep(delay)
                    delay = min(self.max_backoff, delay * 2)
                    continue
                logger.warning(
                    "%s request failed after %d attempt(s): %s",
                    type(self).__name__,
                    self.max_attempts,
                    error,
                )
        return None
