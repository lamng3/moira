from __future__ import annotations
from typing import Any, Dict, List, Optional
from neo4j import Driver
from agentoi.agents.tools.knowledge_graph.config import Config
from agentoi.agents.tools.knowledge_graph.utils.neo4j import build_driver, record_to_item

class Neo4jClient:
    """single-endpoint Neo4j client."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._driver: Optional[Driver] = None
        if cfg.NEO4J_URI:
            self._driver = build_driver(cfg.NEO4J_URI, cfg.NEO4J_USER, cfg.NEO4J_PASSWORD)

        self._db = cfg.NEO4J_DATABASE or None
        self._name_props = getattr(cfg, "NAME_PROPS", ["name"])
        self._desc_props = getattr(cfg, "DESC_PROPS", ["description", "definition"])
        self._iri_props  = getattr(cfg, "IRI_PROPS",  ["url", "uri", "identifier"])
        self._extra_props = getattr(cfg, "EXTRA_PROPS", []) or []

        # allow overriding via catalog key "neo4j" or "default"
        cat = cfg.CYPHER_CATALOG or {}
        self._search_cypher = (cat.get("neo4j", {}) or cat.get("default", {})).get("search")
        self._neigh_cypher  = (cat.get("neo4j", {}) or cat.get("default", {})).get("neighborhood")
        self._random_cypher = (cat.get("neo4j", {}) or cat.get("default", {})).get("random_nodes")
        self._random_labels_cypher = (cat.get("neo4j", {}) or cat.get("default", {})).get("random_nodes_by_labels")

    @property
    def enabled(self) -> bool:
        return self._driver is not None

    def close(self) -> None:
        if self._driver: self._driver.close()

    def search_nodes(self, term: str, *, limit: int = 10) -> List[Dict[str, Any]]:
        if not self.enabled: return []
        q = term.lower()
        items: List[Dict[str, Any]] = []
        assert self._driver is not None
        props = list(dict.fromkeys((self._name_props or []) + self._extra_props + (self._desc_props or [])))
        cypher = self._search_cypher or SEARCH_CYPHER
        with self._driver.session(database=self._db) as sess:
            for r in sess.run(cypher, q=q, name_props=props, limit=limit):
                items.append(record_to_item(
                    r, kg_name="neo4j",
                    name_props=self._name_props, desc_props=self._desc_props, iri_props=self._iri_props
                ))
        return items

    def neighborhood_by_id(self, node_id: int, *, limit_edges: int = 25) -> Dict[str, Any]:
        if not self.enabled:
            return {"nodes": [], "edges": []}
        nodes: Dict[int, Dict[str, Any]] = {node_id: {"id": node_id, "name": f"node/{node_id}", "labels": []}}
        edges: List[Dict[str, Any]] = []
        assert self._driver is not None
        cypher = self._neigh_cypher or NEIGHBORHOOD_CYPHER
        with self._driver.session(database=self._db) as sess:
            for r in sess.run(cypher, id=node_id, limit=limit_edges):
                dst = int(r["dst"])
                props = dict(r["props"])
                nm = props.get("name") or f"node/{dst}"
                nodes[dst] = {"id": dst, "name": nm, "labels": list(r["labels"]), "props": props}
                edges.append({"src": int(r["src"]), "dst": dst, "type": r["rel"]})
        return {"nodes": list(nodes.values()), "edges": edges}
    
    def random_nodes(self, *, limit: int = 10, labels: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Return a random sample of nodes. If `labels` is provided, sampling is restricted to nodes
        having at least one of those labels.
        """
        if not self.enabled:
            return []

        assert self._driver is not None
        cypher = self._random_labels_cypher if labels else self._random_cypher
        params: Dict[str, Any] = {"limit": int(limit)}
        if labels:
            params["labels"] = list(dict.fromkeys(labels))  # de-dup, stable order

        items: List[Dict[str, Any]] = []
        with self._driver.session(database=self._db) as sess:
            for r in sess.run(cypher, **params):
                items.append(
                    record_to_item(
                        r,
                        kg_name="neo4j",
                        name_props=self._name_props,
                        desc_props=self._desc_props,
                        iri_props=self._iri_props,
                    )
                )
        return items