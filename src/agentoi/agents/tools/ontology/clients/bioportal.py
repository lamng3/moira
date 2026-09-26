from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import requests
from agentoi.agents.tools.ontology.http_client import Http
from agentoi.agents.tools.ontology.config import Config

class BioPortalClient:
    def __init__(self, http: Http, cfg: Config):
        self.http, self.cfg = http, cfg

    def search(
        self, 
        term: str, 
        *, 
        ontologies: Optional[List[str]], 
        exact: bool, 
        rows: int
    ) -> Tuple[str, List[Dict[str, Any]]]:
        if not self.cfg.BIOPORTAL_API_KEY:
            return "bioportal-disabled", []
        params = {"q": term, "require_exact_match": str(exact).lower(), "pagesize": rows}
        if ontologies:
            params["ontologies"] = ",".join(ontologies)
        headers = {"Authorization": f"apikey token={self.cfg.BIOPORTAL_API_KEY}"}
        data = self.http.get(
            f"{self.cfg.BIOPORTAL_BASE}/search", 
            params=params, 
            headers=headers
        )
        items = data.get("collection", []) if isinstance(data, dict) else []
        out: List[Dict[str, Any]] = []
        for it in items[:rows]:
            out.append({
                "label": it.get("prefLabel"),
                "synonyms": it.get("synonym") or [],
                "description": (it.get("definition") or [""])[0] if isinstance(it.get("definition"), list) else it.get("definition"),
                "iri": it.get("@id"),
                "short_form": it.get("notation"),
                "ontology_prefix": (it.get("@id") or "").split("/")[-2] if "@id" in it else None,
                "obo_id": it.get("cui") or None,
                "xrefs": it.get("cui") or [],
                "aliases": it.get("altLabel") or [],
                "definitions": it.get("definition") or [],
                "annotations": it.get("properties") or {},
            })
        return "bioportal", out

    def ontology_info(self, acronym: str) -> Optional[Dict[str, Any]]:
        if not self.cfg.BIOPORTAL_API_KEY:
            return None
        headers = {"Authorization": f"apikey token={self.cfg.BIOPORTAL_API_KEY}"}
        meta = self.http.get(
            f"{self.cfg.BIOPORTAL_BASE}/ontologies/{acronym}", 
            params={"include": "all"}, 
            headers=headers
        )
        if not meta or not isinstance(meta, dict):
            return None
        subm = (meta.get("latestSubmission") or {}) if isinstance(meta.get("latestSubmission"), dict) else {}
        metrics = meta.get("metrics") or {}
        return {
            "source": "bioportal",
            "acronym": meta.get("acronym") or acronym,
            "title": meta.get("name") or subm.get("description"),
            "description": subm.get("description") or meta.get("description"),
            "homepage": subm.get("homepage") or meta.get("homepage"),
            "version": subm.get("version") or meta.get("version"),
            "version_iri": subm.get("versionIri") or meta.get("versionIri"),
            "license": subm.get("hasLicense") or meta.get("hasLicense"),
            "created": subm.get("creationDate") or meta.get("creationDate"),
            "updated": subm.get("modificationDate") or meta.get("modificationDate"),
            "metrics": {
                "classes": metrics.get("classes") or metrics.get("numberOfClasses"),
                "properties": metrics.get("properties") or metrics.get("numberOfProperties"),
                "individuals": metrics.get("individuals") or metrics.get("numberOfIndividuals"),
            },
            "raw": meta,
        }