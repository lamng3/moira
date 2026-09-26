from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from agentoi.agents.tools.ontology.http_client import Http
from agentoi.agents.tools.ontology.config import Config

class SPARQLClient:
    def __init__(self, http: Http, cfg: Config):
        self.http, self.cfg = http, cfg

    def _select(self, endpoint: str, query: str) -> Optional[Dict[str, Any]]:
        result = self.http.get(
            endpoint,
            params={"query": query},
            headers=self.cfg.HEADERS_SPARQL,
        )
        return result if isinstance(result, dict) else None

    def dbpedia_info(self) -> Optional[Dict[str, Any]]:
        counts: Dict[str, Optional[int]] = {"classes": None, "properties": None}
        q_classes = (
            "SELECT (COUNT(?c) AS ?n) WHERE { ?c a <http://www.w3.org/2002/07/owl#Class> . "
            'FILTER(STRSTARTS(STR(?c), "http://dbpedia.org/ontology/")) }'
        )
        q_props = (
            "SELECT (COUNT(?p) AS ?n) WHERE { ?p a <http://www.w3.org/1999/02/22-rdf-syntax-ns#Property> . "
            'FILTER(STRSTARTS(STR(?p), "http://dbpedia.org/ontology/")) }'
        )
        for key, q in (("classes", q_classes), ("properties", q_props)):
            js = self._select(self.cfg.DBPEDIA_SPARQL, q)
            try:
                counts[key] = int(js["results"]["bindings"][0]["n"]["value"]) if js else None
            except Exception:
                counts[key] = None
        return {
            "source": "dbpedia-sparql",
            "acronym": "DBpedia",
            "title": "DBpedia Ontology",
            "homepage": "https://www.dbpedia.org/resources/ontology/",
            "metrics": counts,
            "raw": counts,
        }

    def yago_info(self) -> Optional[Dict[str, Any]]:
        q_classes = (
            "SELECT (COUNT(?c) AS ?n) WHERE { ?c a <http://www.w3.org/2002/07/owl#Class> . "
            'FILTER(STRSTARTS(STR(?c), "http://yago-knowledge.org/resource/")) }'
        )
        q_props = (
            "SELECT (COUNT(?p) AS ?n) WHERE { ?p a <http://www.w3.org/1999/02/22-rdf-syntax-ns#Property> . "
            'FILTER(STRSTARTS(STR(?p), "http://yago-knowledge.org/resource/")) }'
        )
        counts: Dict[str, Optional[int]] = {"classes": None, "properties": None}
        for key, q in (("classes", q_classes), ("properties", q_props)):
            js = self._select(self.cfg.YAGO_SPARQL, q)
            try:
                counts[key] = int(js["results"]["bindings"][0]["n"]["value"]) if js else None
            except Exception:
                counts[key] = None
        return {
            "source": "yago-sparql",
            "acronym": "YAGO",
            "title": "YAGO Schema",
            "homepage": "https://yago-knowledge.org/",
            "metrics": counts,
            "raw": counts,
        }

    def wikidata_info(self) -> Optional[Dict[str, Any]]:
        q_props = "SELECT (COUNT(?p) AS ?n) WHERE { ?p a <http://www.wikidata.org/entity/Q18616576> }"
        js = self._select(self.cfg.WIKIDATA_SPARQL, q_props)
        n_props: Optional[int] = None
        try:
            n_props = int(js["results"]["bindings"][0]["n"]["value"]) if js else None
        except Exception:
            pass
        return {
            "source": "wikidata-sparql",
            "acronym": "Wikidata",
            "title": "Wikidata Schema",
            "homepage": "https://www.wikidata.org/",
            "metrics": {"properties": n_props},
            "raw": {"properties": n_props},
        }