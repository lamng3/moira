from __future__ import annotations
from typing import Any, Dict, List
from agentoi.agents.tools.knowledge_graph.http_client import Http
from agentoi.agents.tools.knowledge_graph.config import Config

class DBpediaClient:
    def __init__(self, http: Http, cfg: Config):
        self.http = http
        self.cfg = cfg

    def search(self, term: str, *, limit: int = 10) -> List[Dict[str, Any]]:
        if not self.cfg.ENABLE_DBPEDIA:
            return []
        q = f"""
        SELECT ?s ?label ?abstract WHERE {{
          ?s rdfs:label ?label .
          FILTER (lang(?label)='en') .
          FILTER (CONTAINS(LCASE(STR(?label)), LCASE("{term}"))) .
          OPTIONAL {{ ?s dbo:abstract ?abstract . FILTER(lang(?abstract)='en') }}
        }} LIMIT {int(limit)}
        """
        data = self.http.get_json(self.cfg.DBPEDIA_SPARQL, params={"query": q, "format": "json"})
        out: List[Dict[str, Any]] = []
        for b in data.get("results", {}).get("bindings", []):
            uri = b.get("s", {}).get("value", "")
            label = b.get("label", {}).get("value", "")
            desc = b.get("abstract", {}).get("value", "")
            out.append({
                "source": "dbpedia",
                "uri": uri,
                "name": label,
                "description": desc,
                "labels": ["Thing"],
            })
        return out