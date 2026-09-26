from __future__ import annotations
from typing import Any, Dict, List, Optional
from tavily import TavilyClient as TavilySearch

class TavilyClient:
    def __init__(self, api_key: Optional[str]):
        self._client = TavilySearch(api_key) if (TavilySearch and api_key) else None

    def search(
        self,
        term: str,
        *,
        top_k: int = 5,
        include_domains: Optional[List[str]] = None,
        region: Optional[str] = None,
        time_range: Optional[str] = None,
        max_chars_per_source: int = 800,
    ) -> List[Dict[str, str]]:
        if not self._client:
            return []

        kwargs: Dict[str, Any] = {"max_results": max(1, top_k)}
        if include_domains:
            kwargs["include_domains"] = include_domains
        if region:
            kwargs["location"] = region
        if time_range:
            kwargs["time_range"] = time_range

        try:
            sr = self._client.search(query=term, **kwargs) or {}
        except Exception:
            return []

        out: List[Dict[str, str]] = []
        for r in (sr.get("results") or [])[:top_k]:
            title = (r.get("title") or "").strip() or "(untitled)"
            url = (r.get("url") or "").strip()
            if not url:
                continue
            content = (r.get("content") or "").strip()
            snippet = (f"{title}\n{content}" if content else title)[:max_chars_per_source]
            out.append({"title": title, "url": url, "snippet": snippet})
        return out
