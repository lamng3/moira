from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
from agentoi.agents.tools.common import coerce_terms
from agentoi.agents.tools.website.config import Config
from agentoi.agents.tools.website.http_client import Http
from agentoi.agents.tools.website.clients.tavily import TavilyClient
from agentoi.agents.tools.website.clients.wikipedia import WikipediaClient
from agentoi.agents.tools.website.utils.text import first_sentence, mine_synonyms, top_phrases

class WebsiteLookup:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.http = Http(self.cfg)
        self.tavily = TavilyClient(self.cfg.TAVILY_API_KEY)
        self.wikipedia = WikipediaClient(self.http)

    def term_context(
        self,
        term: Union[str, List[str]],
        *,
        top_k: int = 5,
        site_filters: Optional[List[str]] = None,
        include_wikipedia: bool = True,
        region: Optional[str] = None,
        time_range: Optional[str] = None,
        max_chars: int = 4000,
    ) -> Dict[str, Any]:
        """web search and light ETL to enrich a term."""
        terms = coerce_terms(term)
        per_source = max(200, max_chars // max(1, top_k + (1 if include_wikipedia else 0)))
        result_dict = {}

        for term in terms:
            sources: List[Dict[str, str]] = self.tavily.search(
                term,
                top_k=top_k,
                include_domains=site_filters,
                region=region,
                time_range=time_range,
                max_chars_per_source=per_source,
            )

            wiki_info: Optional[Dict[str, Any]] = None
            if include_wikipedia:
                w = self.wikipedia.get(term)
                if w and (w.get("extract") or w.get("title")):
                    wiki_info = w
                    snippet = (w.get("extract") or "")[:per_source]
                    sources = [{"title": w.get("title") or "Wikipedia",
                                "url": w.get("url") or "",
                                "snippet": snippet}] + sources

            # aggregate bounded text
            blob_parts: List[str] = []
            links: List[Dict[str, str]] = []
            used = 0
            for s in sources:
                part = (s.get("snippet") or "").strip()
                if not part:
                    continue
                remaining = max_chars - used
                if remaining <= 0:
                    break
                if len(part) > remaining:
                    part = part[:remaining]
                blob_parts.append(part)
                used += len(part)
                links.append({"title": s.get("title") or "(untitled)", "url": s.get("url") or ""})

            text_blob = " \n".join(blob_parts)

            # extract features
            synonyms = mine_synonyms(text_blob, term)
            key_terms = top_phrases(text_blob, limit=12, exclude=[term])

            result_dict[term] = {
                "query": {
                    "term": term,
                    "top_k": top_k,
                    "site_filters": site_filters,
                    "include_wikipedia": include_wikipedia,
                    "region": region,
                    "time_range": time_range,
                },
                "summary": first_sentence(text_blob),
                "candidate_synonyms": synonyms,
                "aliases": synonyms,  # backward-compat
                "key_terms": key_terms,
                "sources": links,
                "wikipedia": wiki_info,
            }

        return result_dict

    def close(self) -> None:
        self.http.close()

    def __enter__(self) -> "WebsiteLookup":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

def search_term_context(
    term: Union[str, List[str]],
    top_k: int = 5,
    site_filters: Optional[List[str]] = None,    
    include_wikipedia: bool = True,
    region: Optional[str] = None,
    time_range: Optional[str] = None,
    max_chars: int = 4000
) -> Dict[str, Any]:
    with WebsiteLookup() as lookup:
        return lookup.term_context(
            term=term,
            top_k=top_k,
            site_filters=site_filters,
            include_wikipedia=include_wikipedia,
            region=region,
            time_range=time_range,
            max_chars=max_chars,
        )