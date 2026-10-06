from __future__ import annotations
import os, hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from neo4j import Driver
from moira.agents.tools.common.resources import read_json_resource
from moira.agents.tools.knowledge_graph.utils.neo4j import build_driver, record_to_item
import glob
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class GraphTarget:
    name: str
    uri: str
    user: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None
    name_props: List[str] = field(default_factory=lambda: ["name"])
    extra_props: List[str] = field(default_factory=lambda: ["synonyms","synonym","aliases","alias","altLabel"])
    desc_props: List[str] = field(default_factory=lambda: ["description", "definition"])
    iri_props:  List[str] = field(default_factory=lambda: ["url", "uri", "identifier"])
    search_cypher: Optional[str] = None
    neighborhood_cypher: Optional[str] = None

class Neo4jPool:
    def __init__(self, targets: List[GraphTarget]):
        self._drivers: Dict[str, Tuple[Optional[Driver], GraphTarget]] = {}
        for t in targets:
            drv = build_driver(t.uri, t.user, t.password)
            self._drivers[t.name] = (drv, t)

    @property
    def enabled_names(self) -> List[str]:
        return [name for name, (drv, _) in self._drivers.items() if drv is not None]

    def close(self) -> None:
        for drv, _ in self._drivers.values():
            if drv: drv.close()

    # ---------- n10s admin ----------

    def _require_n10s(self, sess) -> None:
        # Neo4j 5+: use SHOW PROCEDURES
        try:
            rec = sess.run(
                "SHOW PROCEDURES YIELD name "
                "WHERE name STARTS WITH 'n10s.' "
                "RETURN collect(name) AS names"
            ).single()
            names = set(rec["names"] or [])
        except Exception:
            # Neo4j 4.x fallback
            rec = sess.run(
                "CALL dbms.procedures() YIELD name "
                "WHERE name STARTS WITH 'n10s.' "
                "RETURN collect(name) AS names"
            ).single()
            names = set(rec["names"] or [])

        if not names:
            raise RuntimeError(
                "neosemantics (n10s) procedures not found. "
                "Install the n10s plugin matching your Neo4j version and "
                "ensure neo4j.conf includes: dbms.security.procedures.unrestricted=n10s.* "
                "(then restart Neo4j)."
            )

    def ensure_resource_uri_constraint(self, kg_name: str) -> None:
        drv, meta = self._drivers[kg_name]
        if not drv: return
        with drv.session(database=meta.database) as sess:
            # no _require_n10s here
            sess.run("""
                CREATE CONSTRAINT n10s_unique_uri IF NOT EXISTS
                FOR (r:Resource) REQUIRE r.uri IS UNIQUE
            """)

    def init_or_set_graphconfig(self, kg_name: str, opts: Dict[str, Any]) -> None:
        drv, meta = self._drivers[kg_name]
        if not drv: return
        with drv.session(database=meta.database) as sess:
            self._require_n10s(sess)
            try:
                sess.run("CALL n10s.graphconfig.init($o)", o=opts)
            except Exception:
                # already initialized; update instead
                sess.run("CALL n10s.graphconfig.set($o)", o=opts)

    def add_prefixes(self, kg_name: str, prefixes: Dict[str,str]) -> None:
        if not prefixes: return
        drv, meta = self._drivers[kg_name]
        if not drv: return
        with drv.session(database=meta.database) as sess:
            self._require_n10s(sess)
            for pfx, ns in prefixes.items():
                sess.run("CALL n10s.nsprefixes.add($pfx,$ns)", pfx=pfx, ns=ns)

    def clear_resources(self, kg_name: str) -> None:
        """Dangerous: removes previously imported RDF resources."""
        drv, meta = self._drivers[kg_name]
        if not drv: return
        with drv.session(database=meta.database) as sess:
            sess.run("MATCH (n:Resource) DETACH DELETE n")

    def import_turtle_dir(
        self,
        kg_name: str,
        local_turtle_dir: str,
        pattern: str = "*.ttl",
        commit_size: int = 2000,
        language_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        drv, meta = self._drivers[kg_name]
        if not drv: return []
        files = sorted(glob.glob(os.path.join(local_turtle_dir, pattern)))
        out = []
        import_opts: Dict[str, Any] = {"commitSize": int(commit_size)}
        if language_filter:
            import_opts["languageFilter"] = language_filter
        with drv.session(database=meta.database) as sess:
            self._require_n10s(sess)
            for path in files:
                ttl = Path(path).read_text(encoding="utf-8")
                rec = list(sess.run(
                    """
                    CALL n10s.rdf.import.inline($ttl,'Turtle',$opts)
                    YIELD terminationStatus, triplesLoaded, triplesParsed, namespaces, extraInfo
                    RETURN $path AS file, terminationStatus, triplesLoaded, triplesParsed, namespaces, extraInfo
                    """,
                    ttl=ttl, opts=import_opts, path=path
                ))[0]
                out.append({
                    "file": rec["file"],
                    "terminationStatus": rec["terminationStatus"],
                    "triplesLoaded": rec["triplesLoaded"],
                    "triplesParsed": rec["triplesParsed"],
                    "namespaces": rec["namespaces"],
                    "extraInfo": rec["extraInfo"],
                })
        return out

    def search_nodes(self, term: str, *, limit_per_kg: int = 10, only: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        names = [n for n in self.enabled_names if (only is None or n in only)]
        results: List[Dict[str, Any]] = []
        if not names:
            return results
        with ThreadPoolExecutor(max_workers=min(8, len(names))) as ex:
            futs = {ex.submit(self._search_one, n, term, limit_per_kg): n for n in names}
            for f in as_completed(futs):
                try:
                    results.extend(f.result())
                except Exception as e:
                    if os.getenv("KGL_DEBUG") == "1":
                        logger.debug("Knowledge graph pool search failed: %s", e)
        return results

    def _search_one(self, kg_name: str, term: str, limit: int) -> List[Dict[str, Any]]:
        drv, meta = self._drivers[kg_name]
        if not drv:
            return []
        if not meta.search_cypher:
            raise ValueError(f"search_cypher missing for KG '{kg_name}'")
        q = term.lower()
        props = list(dict.fromkeys((meta.name_props or []) + (meta.extra_props or []) + (meta.desc_props or [])))
        cypher = meta.search_cypher
        if os.getenv("KGL_DEBUG") == "1":
            h = hashlib.sha1(cypher.encode()).hexdigest()[:8]
            logger.debug(
                "KG=%s cypher=custom#%s props=%s term=%r",
                kg_name, h, props, q,
            )
        items: List[Dict[str, Any]] = []
        with drv.session(database=meta.database) as sess:
            for r in sess.run(cypher, q=q, name_props=props, limit=limit):
                items.append(record_to_item(
                    r, kg_name=kg_name,
                    name_props=meta.name_props, desc_props=meta.desc_props, iri_props=meta.iri_props
                ))
        return items

    def neighborhood_by_id(self, kg: str, node_id: int, *, limit_edges: int = 25) -> Dict[str, Any]:
        drv, meta = self._drivers.get(kg, (None, None))
        if not drv or not meta:
            return {"nodes": [], "edges": [], "note": f"KG '{kg}' not available"}
        if not meta.neighborhood_cypher:
            return {"nodes": [], "edges": [], "note": f"neighborhood_cypher missing for KG '{kg}'"}
        cypher = meta.neighborhood_cypher
        nodes: Dict[int, Dict[str, Any]] = {node_id: {"id": node_id, "name": f"node/{node_id}", "labels": [], "kg": kg}}
        edges: List[Dict[str, Any]] = []
        with drv.session(database=meta.database) as sess:
            for r in sess.run(cypher, id=node_id, limit=limit_edges):
                dst = int(r["dst"])
                props = dict(r["props"])
                nm = props.get("name") or f"node/{dst}"
                nodes[dst] = {"id": dst, "name": nm, "labels": list(r["labels"]), "props": props, "kg": kg}
                edges.append({"src": int(r["src"]), "dst": dst, "type": r["rel"], "kg": kg})
        return {"kg": kg, "nodes": list(nodes.values()), "edges": edges}

    def ping(self, kg_name: str) -> bool:
        drv, meta = self._drivers.get(kg_name, (None, None))
        if not drv or not meta:
            return False
        try:
            with drv.session(database=meta.database) as sess:
                _ = list(sess.run("RETURN 1 AS ok LIMIT 1"))
            return True
        except Exception:
            return False

    def random_nodes(self, kg: str, *, limit: int = 10, labels: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Return a random sample of nodes from the given pool target (KG name).
        Pulls cypher from cyphers.json via GraphTarget.random_cypher when available,
        otherwise falls back to a default query that supports optional label filtering.
        """
        drv, meta = self._drivers.get(kg, (None, None))
        if not drv or not meta:
            return []

        # Catalog-provided cypher takes precedence (expects $limit and optionally $labels)
        cyphers: Dict[str, Dict[str, str]] = read_json_resource(
            "moira.agents.tools.knowledge_graph",
            "cyphers.json",
        )
        catalog = cyphers.get(kg) or cyphers.get("default", {})
        cypher = catalog.get(
            "random_nodes_by_labels" if labels else "random_nodes"
        )
        if not cypher:
            raise ValueError(f"random_nodes cypher missing for KG '{kg}'")

        params: Dict[str, Any] = {
            "limit": int(limit),
            # Pass empty list when no labels so the fallback WHERE works; custom cyphers can ignore it.
            "labels": list(dict.fromkeys(labels)) if labels else [],
        }

        if os.getenv("KGL_DEBUG") == "1":
            # hash the cypher to avoid dumping it in logs
            h = hashlib.sha1(cypher.encode("utf-8")).hexdigest()[:8]
            logger.debug(
                "random_nodes kg=%s limit=%s labels=%s cypher=#%s",
                kg, params["limit"], params.get("labels"), h,
            )

        items: List[Dict[str, Any]] = []
        with drv.session(database=meta.database) as sess:
            for r in sess.run(cypher, **params):
                items.append(
                    record_to_item(
                        r,
                        kg_name=kg,
                        name_props=meta.name_props,
                        desc_props=meta.desc_props,
                        iri_props=meta.iri_props,
                    )
                )
        return items
