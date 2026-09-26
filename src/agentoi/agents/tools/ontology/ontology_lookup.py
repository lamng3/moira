from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Set, TypeAlias
from agentoi.agents.tools.common import coerce_terms
from agentoi.agents.tools.ontology.http_client import Http
from agentoi.agents.tools.ontology.config import Config
from agentoi.agents.tools.ontology.clients.ols4 import OLS4Client
from agentoi.agents.tools.ontology.clients.bioportal import BioPortalClient
from agentoi.agents.tools.ontology.clients.lov import LOVClient
from agentoi.agents.tools.ontology.clients.sparql import SPARQLClient
from agentoi.agents.tools.ontology.clients.github_rdf import GitHubRDFClient

TestbedName: TypeAlias = str


class OntologyLookup:
    def __init__(
        self, 
        cfg: Optional[Config] = None, 
        testbeds: Optional[Dict[TestbedName, List[str]]] = None
    ):
        self.cfg = cfg or Config()
        self.http = Http(self.cfg)
        self.ols4 = OLS4Client(self.http, self.cfg)
        self.bioportal = BioPortalClient(self.http, self.cfg)
        self.lov = LOVClient(self.http, self.cfg)
        self.sparql = SPARQLClient(self.http, self.cfg)
        self.ghrdf = GitHubRDFClient(self.http, self.cfg)
        self.testbeds = testbeds or self.cfg.DEFAULT_TESTBEDS

    def _search_with_fallback(
        self,
        term: str,
        *,
        ontologies: Optional[List[str]],
        exact: bool,
        rows: int
    ) -> Tuple[str, List[Dict[str, Any]]]:
        try:
            source, results = self.ols4.search(term, ontologies=ontologies, exact=exact, rows=rows)
        except Exception:
            source, results = self.bioportal.search(term, ontologies=ontologies, exact=exact, rows=rows)
        return source, results

    def _trim(self, results: List[Dict[str, Any]], include_fields: List[str]) -> List[Dict[str, Any]]:
        return [{k: r.get(k) for k in include_fields if k in r} for r in results]

    def _dedupe_merge(self, lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Deduplicate across many result lists. Primary key = IRI (if present).
        Fallback key = (ontology_prefix, short_form, label) — stable-ish.
        """
        out: List[Dict[str, Any]] = []
        seen: Set[str] = set()

        def key_of(r: Dict[str, Any]) -> str:
            iri = (r.get("iri") or "").strip().lower()
            if iri:
                return f"iri::{iri}"
            op = (r.get("ontology_prefix") or "").strip().lower()
            sf = (r.get("short_form") or "").strip().lower()
            lb = (r.get("label") or "").strip().lower()
            return f"trip::{op}::{sf}::{lb}"

        for lst in lists:
            for r in lst:
                k = key_of(r)
                if k not in seen:
                    seen.add(k)
                    out.append(r)
        return out

    def term_info(
        self, 
        term: Union[str, List[str]], 
        *, 
        ontologies: Optional[List[str]] = None, 
        exact: bool = False,
        max_results: int = 5, 
        include_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Multi-term aware lookup:
          - If a single term is provided, returns the legacy shape:
              { query: {...}, results: [...], source: "OLS4"|"BioPortal" }
          - If multiple terms are provided, returns:
              {
                query: {..., terms:[...]},
                results: [...deduped merged...],
                by_term: { "<term>": { source: "...", results: [...] }, ... },
                sources: ["OLS4","BioPortal",...],
                count: <len(results)>,
                source: "mixed"  # for backward-compat fields
              }
        """
        include_fields = include_fields or [
            "label", "synonyms", "description", "iri", "short_form",
            "ontology_prefix", "xrefs", "aliases", "definitions", "annotations",
        ]

        terms = coerce_terms(term)
        if len(terms) == 1:
            t = terms[0]
            source, results = self._search_with_fallback(
                t, ontologies=ontologies, exact=exact, rows=max_results
            )
            trimmed = self._trim(results, include_fields)
            return {
                "query": {"term": t, "ontologies": ontologies, "exact": exact, "max_results": max_results},
                "results": trimmed,
                "source": source,
                "count": len(trimmed),
            }

        # multi-term flow
        by_term: Dict[str, Dict[str, Any]] = {}
        per_term_trimmed: List[List[Dict[str, Any]]] = []
        sources: List[str] = []

        for t in terms:
            try:
                src, res = self._search_with_fallback(
                    t, ontologies=ontologies, exact=exact, rows=max_results
                )
            except Exception:
                # last-resort empty
                src, res = ("error", [])
            trimmed = self._trim(res, include_fields)
            by_term[t] = {"source": src, "results": trimmed}
            per_term_trimmed.append(trimmed)
            sources.append(src)

        merged = self._dedupe_merge(per_term_trimmed)
        return {
            "query": {"terms": terms, "ontologies": ontologies, "exact": exact, "max_results": max_results},
            "results": merged,
            "by_term": by_term,
            "sources": sorted(set(sources)),
            "count": len(merged),
            "source": "mixed",  # keep legacy key so downstream code doesn't KeyError
        }

    def _resolve_one(self, name: str) -> Optional[Dict[str, Any]]:
        """aggregate ontology-level metadata"""
        key = name.strip()
        # OLS4
        ols = self.ols4.ontology_info(key.lower()) or self.ols4.ontology_info(key.upper()) or self.ols4.ontology_info(key)
        if ols:
            return ols
        # BioPortal
        bp = self.bioportal.ontology_info(key.upper()) or self.bioportal.ontology_info(key.lower())
        if bp:
            return bp
        # LOV
        lov = self.lov.ontology_info(key.lower())
        if lov:
            return lov
        # named endpoints
        if key.lower() in {"dbpedia", "dbpedia ontology"}:
            return self.sparql.dbpedia_info()
        if key.lower() in {"yago"}:
            return self.sparql.yago_info()
        if key.lower() in {"wikidata", "wiki data", "wikidata schema"}:
            return self.sparql.wikidata_info()
        if key.lower() in {"geooutageonto", "geooutage"}:
            return self.ghrdf.geooutage_info()
        if key.lower() in {"materialinformation", "mi", "mse", "materials ontology", "materials information ontology", "ashino"}:
            return self.ghrdf.materialinformation_info()
        return None

    def ontology_info(self, ontology_or_testbed: str) -> Dict[str, Any]:
        targets = self.testbeds.get(ontology_or_testbed, [ontology_or_testbed])
        items: List[Dict[str, Any]] = []
        for t in targets:
            try:
                info = self._resolve_one(t)
            except Exception:
                info = None
            if info:
                items.append(info)
        return {"query": ontology_or_testbed, "items": items, "count": len(items)}

    def close(self) -> None:
        self.http.close()

    def __enter__(self) -> "OntologyLookup":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
    
def ontology_term_info(
    term: Union[str, List[str]], 
    ontologies: Optional[List[str]] = None,
    exact: bool = False,
    max_results: int = 5,
    include_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Module-level callable used by the agent."""
    with OntologyLookup() as lookup:
        return lookup.term_info(
            term,
            ontologies=ontologies,
            exact=exact,
            max_results=max_results,
            include_fields=include_fields,
        )

def ontology_info(ontology_or_testbed: str) -> Dict[str, Any]:
    with OntologyLookup() as lookup:
        return lookup.ontology_info(ontology_or_testbed)