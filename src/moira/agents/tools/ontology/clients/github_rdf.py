from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import requests
import rdflib
from moira.agents.tools.ontology.http_client import Http
from moira.agents.tools.ontology.config import Config

class GitHubRDFClient:
    def __init__(self, http: Http, cfg: Config):
        self.http, self.cfg = http, cfg

    @staticmethod
    def _parse_rdflib_metadata(rdf_text: str, format_hint: str = "turtle") -> Dict[str, Any]:
        if not rdflib:
            return {}
        g = rdflib.Graph()
        g.parse(data=rdf_text, format=format_hint)
        OWL = rdflib.Namespace("http://www.w3.org/2002/07/owl#")
        DCT = rdflib.Namespace("http://purl.org/dc/terms/")
        VANN = rdflib.Namespace("http://purl.org/vocab/vann/")
        meta: Dict[str, Any] = {}
        for s in g.subjects(rdflib.RDF.type, OWL.Ontology):
            meta["iri"] = str(s)
            meta["title"] = next((str(o) for o in g.objects(s, DCT.title)), None)
            meta["description"] = next((str(o) for o in g.objects(s, DCT.description)), None)
            meta["version"] = next((str(o) for o in g.objects(s, OWL.versionInfo)), None)
            meta["license"] = next((str(o) for o in g.objects(s, DCT.license)), None)
            meta["created"] = next((str(o) for o in g.objects(s, DCT.created)), None)
            meta["modified"] = next((str(o) for o in g.objects(s, DCT.modified)), None)
            meta["preferredNamespacePrefix"] = next((str(o) for o in g.objects(s, VANN.preferredNamespacePrefix)), None)
            meta["preferredNamespaceUri"] = next((str(o) for o in g.objects(s, VANN.preferredNamespaceUri)), None)
            break
        return meta

    def _github_text(self, raw_url: str) -> Optional[str]:
        return self.http.get(raw_url, headers={"Accept": "text/turtle, application/rdf+xml, application/ld+json, */*"})

    def geooutage_info(self, branch: str = "main") -> Optional[Dict[str, Any]]:
        base = f"https://raw.githubusercontent.com/UCF-HENAT/GeoOutageOnto/{branch}"
        for path, fmt in (("/ontology.ttl", "turtle"), ("/ontology.owl", "xml"), ("/ontology.jsonld", "json-ld")):
            txt = self._github_text(base + path)
            if not txt:
                continue
            meta = self._parse_rdflib_metadata(txt, format_hint=fmt)
            if meta:
                return {
                    "source": "github-rdf",
                    "acronym": meta.get("preferredNamespacePrefix") or "GeoOutageOnto",
                    "title": meta.get("title") or "GeoOutageOnto",
                    "description": meta.get("description"),
                    "homepage": "https://ucf-henat.github.io/GeoOutageOnto/",
                    "version": meta.get("version"),
                    "license": meta.get("license"),
                    "created": meta.get("created"),
                    "updated": meta.get("modified"),
                    "metrics": {},
                    "raw": meta,
                }
        return None

    def materialinformation_info(self) -> Optional[Dict[str, Any]]:
        urls: List[str] = []
        if self.cfg.MATERIALINFORMATION_PARTS:
            urls.extend([u.strip() for u in self.cfg.MATERIALINFORMATION_PARTS.split(",") if u.strip()])
        if self.cfg.MATERIALINFORMATION_URL:
            urls.insert(0, self.cfg.MATERIALINFORMATION_URL.strip())
        if not urls:
            return None
        for u in urls:
            txt = self._github_text(u)
            if not txt:
                continue
            fmt = "turtle" if u.endswith((".ttl", ".n3")) else ("json-ld" if u.endswith(".jsonld") else "xml")
            meta = self._parse_rdflib_metadata(txt, format_hint=fmt)
            if meta:
                return {
                    "source": "custom-url",
                    "acronym": meta.get("preferredNamespacePrefix") or "MaterialInformation",
                    "title": meta.get("title") or "Material Information Ontology",
                    "description": meta.get("description"),
                    "homepage": meta.get("preferredNamespaceUri"),
                    "version": meta.get("version"),
                    "license": meta.get("license"),
                    "created": meta.get("created"),
                    "updated": meta.get("modified"),
                    "metrics": {},
                    "raw": meta,
                }
        return None