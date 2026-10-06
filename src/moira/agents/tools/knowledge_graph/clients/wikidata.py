from __future__ import annotations
from typing import Any, Dict, List
from moira.agents.tools.knowledge_graph.http_client import Http
from moira.agents.tools.knowledge_graph.config import Config

class WikidataClient:
    def __init__(self, http: Http, cfg: Config):
        self.http = http
        self.cfg = cfg

    def search(self, term: str, *, limit: int = 10) -> List[Dict[str, Any]]:
        if not self.cfg.ENABLE_WIKIDATA:
            return []
        q = f"""
        SELECT ?item ?itemLabel ?itemDescription ?qid ?rank WHERE {{
            SERVICE wikibase:mwapi {{
                bd:serviceParam wikibase:endpoint "www.wikidata.org";
                                wikibase:api "EntitySearch";
                                mwapi:search "{term}";
                                mwapi:language "en".
                ?item wikibase:apiOutputItem mwapi:item.
                ?rank wikibase:apiOrdinal true.
            }}
            BIND(xsd:integer(STRAFTER(STR(?item), "Q")) AS ?qid)
            SERVICE wikibase:label {{ 
                bd:serviceParam wikibase:language "en". 
                ?item rdfs:label ?itemLabel.
                ?item schema:description ?itemDescription.
            }}
        }}
        ORDER BY ?rank ?qid
        LIMIT {limit}
        """
        data = self.http.get_json(self.cfg.WIKIDATA_SPARQL, params={"query": q, "format": "json"})
        out: List[Dict[str, Any]] = []
        for b in data.get("results", {}).get("bindings", []):
            uri = b.get("item", {}).get("value", "")
            label = b.get("itemLabel", {}).get("value", "")
            desc = b.get("itemDescription", {}).get("value", "")
            out.append({
                "source": "wikidata",
                "uri": uri,
                "name": label,
                "description": desc,
                "labels": ["Item"],
            })
        return out
