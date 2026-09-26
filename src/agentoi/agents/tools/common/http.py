from __future__ import annotations

from typing import Any, Mapping, Optional

import requests


class SessionHttp:
    """Small shared requests session with consistent timeout handling."""

    def __init__(
        self,
        *,
        default_timeout: float,
        headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.default_timeout = default_timeout
        self._s = requests.Session()
        if headers:
            self._s.headers.update(headers)

    def get_json(
        self,
        url: str,
        params: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        response = self._s.get(
            url,
            params=params,
            timeout=timeout or self.default_timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_text(
        self,
        url: str,
        params: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
        accept: Optional[str] = None,
    ) -> str:
        headers = {"Accept": accept} if accept else None
        response = self._s.get(
            url,
            params=params,
            headers=headers,
            timeout=timeout or self.default_timeout,
        )
        response.raise_for_status()
        return response.text

    def post_form(
        self,
        url: str,
        data: Mapping[str, Any],
        headers: Optional[Mapping[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        merged_headers = dict(self._s.headers)
        if headers:
            merged_headers.update(headers)
        response = self._s.post(
            url,
            data=data,
            headers=merged_headers,
            timeout=timeout or self.default_timeout,
        )
        response.raise_for_status()
        return response.json()

    def get(
        self,
        url: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> Optional[Any]:
        """Return JSON/text, preserving ontology providers' fail-soft behavior."""
        try:
            response = self._s.get(
                url,
                params=params,
                headers=headers,
                timeout=self.default_timeout,
            )
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type:
                return response.json()
            return response.text
        except Exception:
            return None

    def close(self) -> None:
        self._s.close()

    def __enter__(self) -> "SessionHttp":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
