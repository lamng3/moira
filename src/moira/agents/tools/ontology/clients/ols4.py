from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import requests
from moira.agents.tools.ontology.http_client import Http
from moira.agents.tools.ontology.config import Config

class OLS4Client:
    def __init__(self, http: Http, cfg: Config):
        self.http, self.cfg = http, cfg

    # term search
    def search(
        self, 
        term: str, 
        *, 
        ontologies: Optional[List[str]], 
        exact: bool, 
        rows: int
    ) -> Tuple[str, List[Dict[str, Any]]]:
        params = {"q": term, "exact": str(exact).lower(), "rows": rows}
        if ontologies:
            params["ontology"] = ",".join(ontologies)
        url = f"{self.cfg.OLS4_BASE}/api/search"
        data = self.http.get(url, params=params)
        docs: List[Dict[str, Any]] = []
        if isinstance(data, dict):
            if isinstance(data.get("response"), dict):
                docs = data["response"].get("docs", [])
            elif isinstance(data.get("docs"), list):
                docs = data["docs"]
            elif isinstance(data.get("results"), list):
                docs = data["results"]
        normalized: List[Dict[str, Any]] = []
        for d in docs[:rows]:
            normalized.append({
                "label": d.get("label") or d.get("label_autosuggest") or d.get("prefLabel"),
                "synonyms": d.get("synonym") or d.get("synonyms") or d.get("obo_synonym") or [],
                "description": (d.get("description") or d.get("definition") or [""])[0]
                                if isinstance(d.get("description") or d.get("definition"), list)
                                else d.get("description") or d.get("definition"),
                "iri": d.get("iri") or d.get("iri_autosuggest") or d.get("@id"),
                "short_form": d.get("short_form") or d.get("shortForm"),
                "ontology_prefix": d.get("ontology_prefix") or d.get("ontology") or d.get("obo_id_prefix"),
                "obo_id": d.get("obo_id") or d.get("oboId"),
                "xrefs": d.get("xref") or d.get("xrefs") or [],
                "aliases": d.get("alias") or d.get("aliases") or [],
                "definitions": d.get("definition") or [],
                "annotations": d.get("annotation") or d.get("annotations") or {},
            })
        return "ols4", normalized

    # ontology-level metadata
    def ontology_info(self, onto: str) -> Optional[Dict[str, Any]]:
        meta = self.http.get(f"{self.cfg.OLS4_BASE}/api/v2/ontologies/{onto}", headers=self.cfg.HEADERS_JSON)
        if not meta:
            meta = self.http.get(f"{self.cfg.OLS4_BASE}/api/ontologies/{onto}", headers=self.cfg.HEADERS_JSON)
        if not meta or not isinstance(meta, dict):
            return None
        metrics = (
            self.http.get(f"{self.cfg.OLS4_BASE}/api/v2/ontologies/{onto}/metrics", headers=self.cfg.HEADERS_JSON)
            or self.http.get(f"{self.cfg.OLS4_BASE}/api/ontologies/{onto}/metrics", headers=self.cfg.HEADERS_JSON)
            or {}
        )
        return {
            "source": "ols4",
            "acronym": meta.get("ontologyId") or meta.get("namespace") or meta.get("preferredPrefix") or onto,
            "title": meta.get("title") or meta.get("config", {}).get("title"),
            "description": meta.get("description") or meta.get("config", {}).get("description"),
            "homepage": meta.get("homepage") or meta.get("config", {}).get("homepage"),
            "version": meta.get("version") or meta.get("config", {}).get("version"),
            "version_iri": meta.get("versionIri") or meta.get("config", {}).get("versionIri"),
            "license": meta.get("license") or meta.get("config", {}).get("license"),
            "created": meta.get("created"),
            "updated": meta.get("updated"),
            "metrics": {
                "classes": metrics.get("classes") or metrics.get("classesCount") or metrics.get("numberOfClasses"),
                "properties": metrics.get("properties") or metrics.get("propertiesCount") or metrics.get("numberOfProperties"),
                "individuals": metrics.get("individuals") or metrics.get("individualsCount") or metrics.get("numberOfIndividuals"),
            },
            "raw": meta,
        }