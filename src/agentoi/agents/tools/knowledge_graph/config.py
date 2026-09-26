from __future__ import annotations
import os, json
import logging
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional, Any

logger = logging.getLogger(__name__)

@dataclass
class Config:
    # Cypher catalog (specified for each KGs)
    NEO4J_CYPHER_JSON: Optional[str] = os.getenv("NEO4J_CYPHER_JSON")
    NEO4J_CYPHER_FILE: Optional[str] = os.getenv("NEO4J_CYPHER_FILE")
    CYPHER_CATALOG: Dict[str, Dict[str, str]] = field(default_factory=dict, init=False)

    # single-endpoint neo4j (set for private graph if needed)
    NEO4J_URI: Optional[str] = os.getenv("NEO4J_URI") # e.g. bolt://localhost:7687 or neo4j+s://<aura-uri>
    NEO4J_USER: Optional[str] = os.getenv("NEO4J_USER")
    NEO4J_PASSWORD: Optional[str] = os.getenv("NEO4J_PASSWORD")
    NEO4J_DATABASE: Optional[str] = os.getenv("NEO4J_DATABASE")

    # multi-endpoint
    NEO4J_TARGETS: Optional[str] = os.getenv("NEO4J_TARGETS")

    # auto-include Hetionet unless disabled
    # todo: do we need this?
    INCLUDE_HETIONET: bool = os.getenv("INCLUDE_HETIONET", "1") != "0"

    # common node property guesses
    NAME_PROPS: List[str] = field(default_factory=lambda: ["name", "label", "prefLabel", "rdfs_label", "preferred_name", "symbol"])
    EXTRA_PROPS: List[str] = field(default_factory=lambda: ["synonyms", "synonym", "aliases", "alias", "altLabel"])
    DESC_PROPS: List[str] = field(default_factory=lambda: ["description", "definition", "abstract", "comment"])
    IRI_PROPS:  List[str] = field(default_factory=lambda: ["iri", "uri", "id", "@id", "uri", "identifier"])

    # HTTP defaults for SPARQL endpoints
    DEFAULT_TIMEOUT: float = float(os.getenv("HTTP_TIMEOUT", 15))
    USER_AGENT: str = os.getenv("USER_AGENT", "kg-lookup/1.0 (+https://example.org)")
    HEADERS_JSON: Dict[str, str] = field(init=False)

    # public KGs
    ENABLE_DBPEDIA: bool = os.getenv("ENABLE_DBPEDIA", "1") != "0"
    ENABLE_WIKIDATA: bool = os.getenv("ENABLE_WIKIDATA", "1") != "0"

    # SPARQL endpoints
    DBPEDIA_SPARQL: str = os.getenv("DBPEDIA_SPARQL", "http://dbpedia.org/sparql")
    WIKIDATA_SPARQL: str = os.getenv("WIKIDATA_SPARQL", "https://query.wikidata.org/sparql")

    JSON_ACCEPT: ClassVar[str] = "application/sparql-results+json"

    # n10s / RDF import settings (all optional; controlled via env)
    NEO4J_IMPORT_DIR: Optional[str] = os.getenv("NEO4J_IMPORT_DIR")  # Neo4j server's 'import' dir (absolute path)
    RDF_TURTLE_DIR: Optional[str] = os.getenv("RDF_TURTLE_DIR")      # local dir that maps INTO NEO4J_IMPORT_DIR
    RDF_TURTLE_GLOB: str = os.getenv("RDF_TURTLE_GLOB", "*.ttl")     # pattern within RDF_TURTLE_DIR
    N10S_GRAPHCONFIG_JSON: Optional[str] = os.getenv("N10S_GRAPHCONFIG_JSON")  # e.g. {"handleVocabUris":"SHORTEN","handleRDFTypes":"LABELS","keepLangTag":true,"handleMultival":"ARRAY"}
    N10S_PREFIXES_JSON: Optional[str] = os.getenv("N10S_PREFIXES_JSON", """{"rdfs":"http://www.w3.org/2000/01/rdf-schema#","skos":"http://www.w3.org/2004/02/skos/core#","schema":"http://schema.org/"}""")
    N10S_IMPORT_COMMIT_SIZE: int = int(os.getenv("N10S_IMPORT_COMMIT_SIZE", "200"))
    N10S_IMPORT_LANGUAGE_FILTER: Optional[str] = os.getenv("N10S_IMPORT_LANGUAGE_FILTER")  # e.g. '["en"]'
    CLEAR_BEFORE_IMPORT: bool = os.getenv("CLEAR_BEFORE_IMPORT", "0") == "1"

    def __post_init__(self):
        self.HEADERS_JSON = {"Accept": self.JSON_ACCEPT, "User-Agent": self.USER_AGENT}
        self.CYPHER_CATALOG = self._load_cypher_catalog()

    def _load_cypher_catalog(self) -> Dict[str, Dict[str, str]]:
        cat: Dict[str, Dict[str, str]] = {}
        DEBUG = os.getenv("KGL_DEBUG") == "1"

        # load from file if provided
        path = self.NEO4J_CYPHER_FILE
        if path:
            try:
                import os.path as p
                path = p.expandvars(p.expanduser(path))
                if not p.isabs(path):
                    path = p.abspath(path)
                if p.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        for k, v in data.items():
                            if isinstance(v, dict):
                                cat[str(k)] = {kk: vv for kk, vv in v.items() if isinstance(kk, str) and isinstance(vv, str)}
                    elif DEBUG:
                        logger.debug("Cypher file is not a dict: %s at %s", type(data), path)
                elif DEBUG:
                    logger.debug("Cypher file not found: %s", path)
            except Exception as e:
                if DEBUG:
                    logger.debug("Cypher file load failed: %s", e)

        # inline json
        inline = self.NEO4J_CYPHER_JSON
        if inline:
            try:
                data = json.loads(inline)
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, dict):
                            base = cat.get(str(k), {})
                            base.update({kk: vv for kk, vv in v.items() if isinstance(kk, str) and isinstance(vv, str)})
                            cat[str(k)] = base
                elif DEBUG:
                    logger.debug("Inline cypher is not a dict: %s", type(data))
            except Exception as e:
                if DEBUG:
                    logger.debug("Inline cypher parse failed: %s", e)

        return cat

    def build_neo4j_targets(self) -> List[Dict[str, Any]]:
        targets: List[Dict[str, Any]] = []

        # prefer explicit JSON env
        if self.NEO4J_TARGETS:
            try:
                data = json.loads(self.NEO4J_TARGETS)
                if isinstance(data, list):
                    for t in data:
                        if not isinstance(t, dict) or "name" not in t or "uri" not in t:
                            continue
                        targets.append(self._apply_target_defaults(t))
            except Exception:
                # ignore parse errors
                pass

        # single endpoint
        if self.NEO4J_URI and not any(t.get("uri") == self.NEO4J_URI for t in targets):
            targets.append(self._apply_target_defaults({
                "name": "neo4j",
                "uri": self.NEO4J_URI,
                "user": self.NEO4J_USER,
                "password": self.NEO4J_PASSWORD,
                "database": self.NEO4J_DATABASE,
            }))

        # auto-include Hetionet unless disabled or already present
        if self.INCLUDE_HETIONET and not any(t.get("name") == "hetionet" for t in targets):
            targets.append(self._apply_target_defaults({
                "name": "hetionet",
                "uri": "bolt://neo4j.het.io",
                "database": None,
                "name_props": ["name", "preferred_name", "symbol"],
                "desc_props": ["description", "definition"],
                "iri_props":  ["url", "uri", "identifier"],
            }))

        # inject per-KG cyphers from catalog
        default = self.CYPHER_CATALOG.get("default", {})
        for t in targets:
            name = t.get("name", "")
            cat = self.CYPHER_CATALOG.get(name, {})
            t.setdefault("search_cypher",      cat.get("search")      or default.get("search"))
            t.setdefault("neighborhood_cypher",cat.get("neighborhood")or default.get("neighborhood"))
                
        return targets

    def _apply_target_defaults(self, t: Dict[str, Any]) -> Dict[str, Any]:
        # fill missing prop lists with global defaults
        t.setdefault("name_props", self.NAME_PROPS)
        t.setdefault("extra_props", self.EXTRA_PROPS)
        t.setdefault("desc_props", self.DESC_PROPS)
        t.setdefault("iri_props",  self.IRI_PROPS)
        return t
    
    # -------- helpers for n10s --------
    def n10s_graphconfig(self) -> Dict[str, Any]:
        base = {"handleVocabUris":"SHORTEN","handleRDFTypes":"LABELS","keepLangTag":True,"handleMultival":"ARRAY"}
        if self.N10S_GRAPHCONFIG_JSON:
            try:
                base.update(json.loads(self.N10S_GRAPHCONFIG_JSON))
            except Exception:
                pass
        return base

    def n10s_prefixes(self) -> Dict[str, str]:
        if not self.N10S_PREFIXES_JSON:
            return {}
        try:
            data = json.loads(self.N10S_PREFIXES_JSON)
            return {str(k): str(v) for k,v in data.items()}
        except Exception:
            return {}

    def n10s_language_filter(self) -> Optional[List[str]]:
        if not self.N10S_IMPORT_LANGUAGE_FILTER:
            return None
        try:
            vals = json.loads(self.N10S_IMPORT_LANGUAGE_FILTER)
            if isinstance(vals, list):
                return [str(x) for x in vals]
        except Exception:
            pass
        return None