from __future__ import annotations
from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from agentoi.agents.tools.common import coerce_terms
from agentoi.agents.tools.knowledge_graph.http_client import Http
from agentoi.agents.tools.knowledge_graph.config import Config
from agentoi.agents.tools.knowledge_graph.utils.text import first_sentence, top_terms
from agentoi.agents.tools.knowledge_graph.clients.neo4j import Neo4jClient
from agentoi.agents.tools.knowledge_graph.clients.neo4jpool import Neo4jPool, GraphTarget
from agentoi.agents.tools.knowledge_graph.clients.dbpedia import DBpediaClient
from agentoi.agents.tools.knowledge_graph.clients.wikidata import WikidataClient

import os
DEBUG = os.getenv("KGL_DEBUG", "0") == "1"
logger = logging.getLogger(__name__)

@dataclass
class SearchOpts:
    top_k: int = 10
    include_public: bool = True
    only_kgs: Optional[List[str]] = None
    max_merged: int = 100

class KnowledgeGraphLookup:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.http = Http(self.cfg)

        # multi-endpoint pool
        targets_data = self.cfg.build_neo4j_targets()
        targets: List[GraphTarget] = [GraphTarget(**t) for t in targets_data]
        self.pool = Neo4jPool(targets)

        # back-compat single endpoint
        self.neo4j = Neo4jClient(self.cfg)

        # public KGs
        self.dbpedia = DBpediaClient(self.http, self.cfg)
        self.wikidata = WikidataClient(self.http, self.cfg)

    def bootstrap_import_from_turtle(self, kg: str = "neo4j") -> List[Dict[str, Any]]:
        """
        1) Ensure :Resource(uri) constraint
        2) Init/Set n10s graphconfig
        3) Add prefixes
        4) (Optional) Clear existing :Resource
        5) Import all *.ttl files from RDF_TURTLE_DIR inside NEO4J_IMPORT_DIR
        """
        if kg not in self.pool.enabled_names:
            raise ValueError(f"KG '{kg}' is not available. Enabled: {self.pool.enabled_names}")

        # 1
        self.pool.ensure_resource_uri_constraint(kg)
        # 2
        self.pool.init_or_set_graphconfig(kg, self.cfg.n10s_graphconfig())
        # 3
        self.pool.add_prefixes(kg, self.cfg.n10s_prefixes())
        # 4
        if self.cfg.CLEAR_BEFORE_IMPORT:
            self.pool.clear_resources(kg)
        # 5
        if not (self.cfg.NEO4J_IMPORT_DIR and self.cfg.RDF_TURTLE_DIR):
            raise ValueError("NEO4J_IMPORT_DIR and RDF_TURTLE_DIR must be set to import Turtle files.")
        return self.pool.import_turtle_dir(
            kg,
            local_turtle_dir=self.cfg.RDF_TURTLE_DIR,
            pattern=self.cfg.RDF_TURTLE_GLOB,
            commit_size=self.cfg.N10S_IMPORT_COMMIT_SIZE,
            language_filter=self.cfg.n10s_language_filter()
        )
    
    def list_kgs(self) -> List[str]:
        return self.pool.enabled_names + (["neo4j"] if self.neo4j.enabled else [])

    def entity_search(
        self,
        term: Union[str, List[str]],
        *,
        top_k: Optional[int] = None,
        include_public: Optional[bool] = None,
        only_kgs: Optional[List[str]] = None,
        max_merged: Optional[int] = None,
        opts: Optional[SearchOpts] = None,
    ) -> Dict[str, Any]:
        """search entities across configured KGs and return merged results."""
        if opts is None:
            opts = SearchOpts(
                top_k=top_k if top_k is not None else 10,
                include_public=True if include_public is None else include_public,
                only_kgs=only_kgs,
                max_merged=max_merged if max_merged is not None else 100,
            )
        result_dict = {}
        terms = coerce_terms(term)

        for term in terms:
            items: List[Dict[str, Any]] = []
            # neo4j multi-endpoint
            try:
                items.extend(self.pool.search_nodes(term, limit_per_kg=opts.top_k, only=opts.only_kgs))
            except Exception as e:
                if DEBUG:
                    logger.debug("Neo4j pool search failed: %s", e)

            # single endpoint
            if self.neo4j.enabled and (opts.only_kgs is None or "neo4j" in opts.only_kgs):
                try:
                    neo_items = self.neo4j.search_nodes(term, limit=opts.top_k)
                    for it in neo_items:
                        it.setdefault("kg", "neo4j")
                    items.extend(neo_items)
                except Exception as e:
                    if DEBUG:
                        logger.debug("Neo4j single search failed: %s", e)

            # public KGs
            if opts.include_public:
                try:
                    items.extend(self.dbpedia.search(term, limit=opts.top_k))
                except Exception as e:
                    if DEBUG:
                        logger.debug("DBpedia search failed: %s", e)
                try:
                    items.extend(self.wikidata.search(term, limit=opts.top_k))
                except Exception as e:
                    if DEBUG:
                        logger.debug("Wikidata search failed: %s", e)

            merged = self._merge_rank(term, items, max_merged=opts.max_merged)
            result_dict[term] = {"query": {"term": term, **vars(opts)}, "count": len(merged), "items": merged}

        return result_dict

    def neighborhood(self, *, kg: Optional[str] = None, neo4j_id: Optional[int] = None, limit_edges: int = 25) -> Dict[str, Any]:
        if neo4j_id is None:
            return {"nodes": [], "edges": [], "note": "neo4j_id is required"}
        if kg:
            return self.pool.neighborhood_by_id(kg, neo4j_id, limit_edges=limit_edges)
        return self.neo4j.neighborhood_by_id(neo4j_id, limit_edges=limit_edges)
    
    def random_nodes(
        self,
        *,
        kg: Optional[str] = None,
        limit: int = 10,
        labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Return a random sample of nodes from the specified KG.
        - If `kg` is None or "neo4j", use the single-endpoint Neo4j client (if enabled).
        - If `kg` is set to one of the multi-endpoint pool names, this will attempt to use
          a pool implementation (if available). Otherwise returns a helpful note.
        """
        # Prefer explicit KG in the pool when given
        if kg and kg in self.pool.enabled_names:
            try:
                items = self.pool.random_nodes(kg, limit=limit, labels=labels)  # type: ignore[attr-defined]
                return {"kg": kg, "count": len(items), "items": items}
            except AttributeError:
                if DEBUG:
                    logger.debug("pool.random_nodes not implemented for kg=%s", kg)
                # Fallback to single-endpoint
                if self.neo4j.enabled and (kg is None or kg == "neo4j"):
                    items = self.neo4j.random_nodes(limit=limit, labels=labels)
                    for it in items:
                        it.setdefault("kg", "neo4j")
                    return {"kg": "neo4j", "count": len(items), "items": items}

        return {"kg": kg or "neo4j", "count": 0, "items": [], "note": "No available Neo4j endpoint"}

    def close(self) -> None:
        self.http.close()
        self.neo4j.close()
        self.pool.close()

    def __enter__(self) -> "KnowledgeGraphLookup":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _merge_rank(self, term: str, items: List[Dict[str, Any]], *, max_merged: int) -> List[Dict[str, Any]]:
        seen: Dict[Tuple[str, str], int] = {}
        uniq: List[Dict[str, Any]] = []
        t = term.lower()

        for it in items:
            # normalize summary/keywords
            desc = (it.get("description") or "")[:2000]
            if desc and not it.get("summary"):
                it["summary"] = first_sentence(desc, limit=240)
            if desc and not it.get("keywords"):
                it["keywords"] = top_terms(desc, limit=8)

            # dedup key priority: (source, iri/uri) -> (source, id) -> (source, name)
            src = it.get("source", "?")
            key_val = (it.get("iri") or it.get("uri") or str(it.get("id") or "") or str(it.get("name") or "")).lower()
            dkey = (src, key_val)
            if dkey in seen:
                j = seen[dkey]
                uniq[j] = self._merge_two(uniq[j], it)
                continue
            seen[dkey] = len(uniq)
            uniq.append(it)

        uniq.sort(key=lambda x: self._score(t, x), reverse=True)
        return uniq[:max_merged]

    @staticmethod
    def _merge_two(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        for k in ("summary", "description"):
            if not a.get(k) and b.get(k):
                a[k] = b[k]
        if not a.get("keywords") and b.get("keywords"):
            a["keywords"] = b["keywords"]
        return a

    @staticmethod
    def _score(t: str, it: Dict[str, Any]) -> float:
        name = str(it.get("name") or "").lower()
        base = 0.0
        if name == t:
            base += 10.0
        elif name.startswith(t):
            base += 6.0
        elif t in name:
            base += 3.0
        # prefer structured/owned KGs slightly
        if it.get("source") == "neo4j":
            base += 1.0
        kg = (it.get("kg") or "").lower()
        if kg and kg not in {"hetionet", "neo4j"}:
            base += 0.5
        return base

def search_knowledge_graph(
    term: Union[str, List[str]],
    top_k: int = 5,
    include_public: Optional[bool] = None,
    only_kgs: Optional[List[str]] = None,
    max_merged: Optional[int] = None,
    opts: Optional[SearchOpts] = None
) -> Dict[str, Any]:
    with KnowledgeGraphLookup() as lookup:
        return lookup.entity_search(
            term=term,
            top_k=top_k,
            include_public=include_public,
            only_kgs=only_kgs,
            max_merged=max_merged,
            opts=opts,
        )