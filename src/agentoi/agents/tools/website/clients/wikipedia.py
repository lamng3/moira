from __future__ import annotations
from typing import Any, Dict, Optional
import requests

from ..http_client import Http

class WikipediaClient:
    def __init__(self, http: Http, timeout: float = 10.0):
        self._http = http
        self._timeout = timeout

    def get(self, term: str) -> Optional[Dict[str, Any]]:
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(term)}"
            data = self._http.get_json(url, timeout=self._timeout) or {}
            title = data.get("title")
            extract = data.get("extract")
            url_page = (data.get("content_urls") or {}).get("desktop", {}).get("page")
            if not (title or extract):
                return None
            return {"title": title, "extract": extract, "url": url_page}
        except Exception:
            return None
