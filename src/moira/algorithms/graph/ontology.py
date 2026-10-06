from __future__ import annotations

import bisect
import hashlib
import math
import os
import pickle
import random
import re
import time
import uuid
from collections import defaultdict
from typing import Any

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS, XSD

from moira import legacy_pickle
from .concept import Concept, ConceptRelation
from .identifiers import name_from_uuid
from .noise import BARTNoiser, bart_default_config
from .predicates import is_hierarchy_predicate


class Ontology:
    """ontology = (nodes, edges)"""

    def __init__(
        self,
        nodes: list[Concept] | None = None,
        edges: list[ConceptRelation] | None = None,
        version: str | None = None,
        name: str | None = None,
        node_map: dict[str, Concept] | None = None,
    ):
        self.nodes: list[Concept] = nodes if nodes is not None else []
        self.edges: list[ConceptRelation] = edges if edges is not None else []
        self.version: str = version if version is not None else "0.1.0"
        # name_from_uuid(u, "slug")     = "q1x7c9DkQ7u3e6Qv1bZb9Q"
        # name_from_uuid(u, "codename") = "brisk-otter-a3f9"
        self.name: str = (
            name if name is not None else name_from_uuid(uuid.uuid4(), "codename")
        )
        self.node_map: dict[str, Concept] = node_map if node_map is not None else {}
        # lazy build index
        self._local_index_built = False
        self._local_index: defaultdict[str, list[str]] = defaultdict(list)

    def _clear(self):
        self.nodes = []
        self.edges = []

    @staticmethod
    def _strip_angles(s: str) -> str:
        s = s.strip()
        return s[1:-1] if s.startswith("<") and s.endswith(">") else s

    @staticmethod
    def _local_part_from_any(s: str) -> str:
        s = Ontology._strip_angles(s)
        if "://" in s:  # absolute IRI
            pos = max(s.rfind("#"), s.rfind("/"))
            return s[pos + 1 :] if pos != -1 else s
        if ":" in s and not s.startswith("_:"):  # CURIE
            return s.split(":", 1)[1]
        return s  # bare

    def _rebuild_local_index(self):
        self._local_index.clear()
        for key in self.node_map:
            loc = self._local_part_from_any(key)
            self._local_index[loc].append(key)
        # tie-breaking
        for loc in self._local_index:
            self._local_index[loc].sort()
        self._local_index_built = True

    def _resolve_lookup_key(self, s: str) -> str:
        s = (s or "").strip()
        if not s:
            return s
        s = self._strip_angles(s)
        if s in self.node_map:
            return s
        if "://" in s:
            return s
        # requires local-name index
        if not self._local_index_built:
            self._rebuild_local_index()
        local = self._local_part_from_any(s)
        cands = self._local_index.get(local, [])
        if not cands:
            return s

        # prefer IRIs if CURIE given
        if ":" in s and not s.startswith("_:"):
            pref = s.split(":", 1)[0].lower()
            hinted = [iri for iri in cands if pref in iri.lower()]
            if hinted:
                return hinted[0]

        # fallback: first deterministic candidate
        return cands[0]

    def get_node(self, key: str) -> Concept:
        """get specified node"""
        norm = self._resolve_lookup_key(key)
        if norm in self.node_map:
            return self.node_map[norm]

        # create a new node on miss
        nodeid = "N" + hashlib.md5(norm.encode("utf-8")).hexdigest()

        iri = norm if "://" in norm else None
        c = Concept(name=key, nodeid=nodeid, iri=iri)
        self.node_map[norm] = c

        if self._local_index_built:
            loc = self._local_part_from_any(norm)
            # insert keeping list sorted for determinism
            lst = self._local_index.setdefault(loc, [])
            bisect.insort(lst, norm)

        return c

    def build_ontology_from_triples(self, triples: list[tuple[str, str, str]]):
        """build ontology from a set of triples (reset ontology)"""
        if triples is None:
            raise ValueError("Unable to build ontology; No triples received")

        # reset
        self._clear()

        # fast lookup (reset node_map)
        self.node_map: dict[str, Concept] = {}
        edge_seen: set[tuple[str, str, str]] = set()

        for s, p, o in triples:
            s = (s or "").strip()
            p = (p or "").strip()
            o = (o or "").strip()
            if not s or not p or not o:
                continue

            src = self.get_node(s)
            tgt = self.get_node(o)
            key = (s, p, o)
            if key in edge_seen:
                continue
            edge_seen.add(key)

            self.edges.append(ConceptRelation(src, p, tgt))

            # update each concept with its parents and children (by predicate type)
            # record both directions
            src.add_relation(p, tgt, direction="out")
            tgt.add_relation(p, src, direction="in")

        self.nodes = list(self.node_map.values())
        self._rebuild_local_index()
        return self

    def extract_dense_subontology(
        self,
        max_nodes: int,
        max_edges: int | None = None,
    ) -> Ontology:
        """sample dense subontology of ontology"""
        if max_nodes <= 0 or not self.nodes:
            return Ontology(
                nodes=[], edges=[], version=self.version, name=f"{self.name}-empty"
            )

        # degree
        in_deg = defaultdict(int)
        out_deg = defaultdict(int)
        for e in self.edges:
            out_deg[e.src] += 1
            in_deg[e.tgt] += 1

        # top-K by total directed degree
        sel_nodes = sorted(
            self.nodes,
            key=lambda n: (
                in_deg.get(n, 0) + out_deg.get(n, 0),
                out_deg.get(n, 0),
                n.name,
            ),
            reverse=True,
        )[:max_nodes]
        sel_set = set(sel_nodes)

        # induce directed edges
        sub_edges = [e for e in self.edges if e.src in sel_set and e.tgt in sel_set]

        # trim while keeping direction (not creating cycles)
        if max_edges is not None and len(sub_edges) > max_edges:

            def _is_hierarchy(pred: str) -> bool:
                # Concept predicate normalization
                dummy = sel_nodes[0]
                return dummy.is_is_a_predicate(pred)

            def _edge_key(e: ConceptRelation):
                isa = 1 if _is_hierarchy(e.pred) else 0
                return (isa, getattr(e, "score", 0.0))

            sub_edges.sort(key=_edge_key, reverse=True)
            sub_edges = sub_edges[:max_edges]

        new_name = f"{self.name}"
        return Ontology(
            nodes=sel_nodes, edges=sub_edges, version=self.version, name=new_name
        )

    def dense_subontology(self, *args, **kwargs) -> Ontology:
        return self.extract_dense_subontology(*args, **kwargs)

    def extract_dense_subontology_random(
        self,
        max_nodes: int,
        max_edges: int | None = None,
        *,
        seed: int | None = None,
        noise: float = 0.0,  # 0 = deterministic; >0 add noise
        exclude_nodes: set[str] | None = None,  # set of nodeids to avoid
    ) -> Ontology:
        """sample dense subontology of ontology with randomness"""
        if max_nodes <= 0 or not self.nodes:
            return Ontology(
                nodes=[], edges=[], version=self.version, name=f"{self.name}-empty"
            )

        rng = random.Random(seed)

        # degrees
        in_deg, out_deg = defaultdict(int), defaultdict(int)
        for e in self.edges:
            out_deg[e.src] += 1
            in_deg[e.tgt] += 1

        # candidates with node exclusion for diversity)
        excl = exclude_nodes or set()
        cands = [n for n in self.nodes if n.nodeid not in excl] or list(self.nodes)

        def gumbel():
            u = max(1e-12, rng.random())
            return -math.log(-math.log(u))

        def score(n: Concept):
            base = in_deg.get(n, 0) + out_deg.get(n, 0)
            return base + (noise * gumbel() if noise > 0 else 0.0)

        sel_nodes = sorted(cands, key=score, reverse=True)[:max_nodes]
        sel_set = set(sel_nodes)

        # induce directed edges
        sub_edges = [e for e in self.edges if e.src in sel_set and e.tgt in sel_set]

        # prioritize hierarchy edges, then by edge.score
        if max_edges is not None and len(sub_edges) > max_edges:
            # use Concept.is_is_a_predicate for selected node
            dummy = sel_nodes[0]

            def edge_key(e: ConceptRelation):
                isa = 1 if dummy.is_is_a_predicate(e.pred) else 0
                jitter = rng.random() * 1e-6  # stable but breaks ties
                return (isa, getattr(e, "score", 0.0), jitter)

            sub_edges.sort(key=edge_key, reverse=True)
            sub_edges = sub_edges[:max_edges]

        new_name = f"{self.name}"
        return Ontology(
            nodes=sel_nodes, edges=sub_edges, version=self.version, name=new_name
        )

    def dense_subontology_random(self, *args, **kwargs) -> Ontology:
        return self.extract_dense_subontology_random(*args, **kwargs)

    def update_ontology_with_new_triples(
        self, triples: list[tuple[str, str, str]]
    ) -> Ontology:
        """Add valid, non-duplicate triples without rebuilding the ontology."""
        if triples is None:
            raise ValueError("Unable to update ontology; no triples received")

        existing = {
            (edge.src.nodeid, edge.pred, edge.tgt.nodeid) for edge in self.edges
        }
        for subject, predicate, object_ in triples:
            subject = (subject or "").strip()
            predicate = (predicate or "").strip()
            object_ = (object_ or "").strip()
            if not subject or not predicate or not object_:
                continue
            source = self.get_node(subject)
            target = self.get_node(object_)
            key = (source.nodeid, predicate, target.nodeid)
            if key in existing:
                continue
            existing.add(key)
            self.edges.append(ConceptRelation(source, predicate, target))
            source.add_relation(predicate, target, direction="out")
            target.add_relation(predicate, source, direction="in")

        self.nodes = list(self.node_map.values())
        for node in self.nodes:
            node.rank = None
        self._rebuild_local_index()
        return self

    @staticmethod
    def copy_ground_set(gs: dict[str, list[str]]) -> dict[str, list[str]]:
        return {k: list(v) for k, v in (gs or {}).items()}

    def _key_of(self, c: Concept) -> str:
        # Return the node_map key for this Concept if present, otherwise resolve from name/iri
        for k, v in self.node_map.items():
            if v is c:
                return k
        # fallback to existing resolvers
        hint = c.iri or c.name
        return (
            self._resolve_lookup_key(hint)
            if hint
            else next(iter(self.node_map.keys()), "")
        )

    def apply_bart_noise(
        self, *, cfg: dict[str, Any] | None = None, seed: int | None = None
    ) -> tuple[Ontology, dict[str, Any]]:
        """
        Clone the ontology WITHOUT calling get_node()/resolver:
        - Preserve exact node_map keys (IRI/CURIE), nodeid, name, iri, ground_set.
        - Recreate edges using the same keys.
        - Then inject BART-style noise into textual fields.
        """
        _cfg = {**bart_default_config(), **(cfg or {})}
        noiser = BARTNoiser(_cfg, seed=seed)
        rng = noiser.rng

        # --- 1) Make a blank target and copy header ---
        noisy = Ontology(
            nodes=[], edges=[], version=self.version, name=f"{self.name}-bart"
        )
        noisy.node_map = {}
        noisy._local_index_built = False
        noisy._local_index = defaultdict(list)

        # Reverse map: Concept -> key used in this ontology
        concept_to_key: dict[Concept, str] = {c: k for k, c in self.node_map.items()}

        # --- 2) Clone nodes EXACTLY (preserve key, nodeid, name, iri, ground_set) ---
        for key, orig in self.node_map.items():
            clone = Concept(
                name=orig.name,
                nodeid=orig.nodeid,  # preserve stable id
                iri=orig.iri,
                ground_set=self.copy_ground_set(orig.ground_set),
            )
            # parents/children/out_rels/in_rels will be rebuilt from edges
            noisy.node_map[key] = clone

        # Establish .nodes list and local-name index (based on exact keys)
        noisy.nodes = list(noisy.node_map.values())
        noisy._rebuild_local_index()

        # --- 3) Recreate edges using the exact original keys (no resolution) ---
        noisy.edges = []
        for e in self.edges:
            skey = concept_to_key.get(e.src)
            tkey = concept_to_key.get(e.tgt)
            if skey is None or tkey is None:
                # Shouldn't happen; skip defensively
                continue
            s = noisy.node_map[skey]
            t = noisy.node_map[tkey]
            new_e = ConceptRelation(s, e.pred, t, score=getattr(e, "score", 0.0))
            noisy.edges.append(new_e)
            # Maintain adjacency + parents/children via predicate type
            s.add_relation(e.pred, t, direction="out")
            t.add_relation(e.pred, s, direction="in")

        # --- 4) Apply BART noise to textual fields in the CLONE ---
        meta_noised_nodes: set[str] = set()
        meta_field_changes: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        for key, tgt in noisy.node_map.items():
            if not noiser._maybe(_cfg["p_record"]):
                continue
            meta_noised_nodes.add(tgt.nodeid)

            # fields to consider
            for field in _cfg["fields"]:
                vals = tgt.ground_set.get(field, [])
                if not vals:
                    continue
                new_vals = noiser.noise_list(vals)
                changed = sum(1 for a, b in zip(vals, new_vals) if a != b)
                if changed:
                    meta_field_changes[tgt.nodeid][field] += changed
                tgt.ground_set[field] = new_vals

            # (rare) also corrupt Concept.name if requested
            if _cfg.get("p_mutate_name", 0) > 0 and noiser._maybe(
                _cfg["p_mutate_name"]
            ):
                new_name = noiser.noise_string(tgt.name or "")
                if new_name and new_name != tgt.name:
                    tgt.name = new_name

        # --- 5) Optional: rare cross-record 'labels' swap (value confusion) ---
        swaps: list[tuple[str, str]] = []
        if _cfg.get("p_label_swap", 0) > 0 and rng.random() < _cfg["p_label_swap"]:
            cand_nodes = [
                n for n in noisy.node_map.values() if n.ground_set.get("labels")
            ]
            if len(cand_nodes) >= 2:
                a, b = rng.sample(cand_nodes, 2)
                a.ground_set["labels"], b.ground_set["labels"] = (
                    b.ground_set.get("labels", []),
                    a.ground_set.get("labels", []),
                )
                swaps.append((a.nodeid, b.nodeid))

        meta = {
            "noised_nodes": meta_noised_nodes,
            "field_changes": meta_field_changes,
            "label_swaps": swaps,
            "config": _cfg,
            "seed": seed,
        }
        return noisy, meta

    def to_dict(self) -> dict:
        """snapshot of an ontology"""
        nodes = {n.name: n.nodeid for n in self.nodes}
        return {
            "name": self.name,
            "version": self.version,
            "meta": {
                "saved_at": int(time.time()),
            },
            "nodes": [{"id": nodes[c.name], "label": c.name} for c in self.nodes],
            "edges": [
                {"src": nodes[e.src.name], "pred": e.pred, "tgt": nodes[e.tgt.name]}
                for e in self.edges
            ],
        }

    def to_rdf(
        self,
        dirpath: str | None = None,
        filename: str | None = None,
        format: str = "turtle",
        base_iri: tuple[str, str] | None = None,
        prefixes: dict[str, str] | None = None,
        declare_properties: bool = True,
        include_labels: bool = True,
    ) -> str:
        """
        Serialize the current ontology to RDF (Turtle or RDF/XML/OWL) and save to disk.

        Args:
            filepath: Full path to write to. If None, writes to 'data/ontologies/{version}-{name}.(ttl|owl)'.
            format: 'turtle' (or 'ttl') for Turtle, 'xml'/'rdfxml'/'owl' for RDF/XML/OWL.
            base_iri: Tuple containing Base IRI for minting terms (when a Concept lacks .iri or a predicate isn't an absolute IRI) and prefix for Base IRI.
                      Defaults to (f'http://example.org/{self.name}#', 'ont').
            prefixes: Optional dict of CURIE prefixes, e.g., {'ex':'http://example.org/', 'skos':'http://www.w3.org/2004/02/skos/core#'}
                      Used to expand CURIE predicates like 'ex:partOf'. Bound to the graph for nicer output.
            declare_properties: If True, declare non 'is-a' predicates as owl:ObjectProperty.
            include_labels: If True, add rdfs:label to each class with the Concept.name.

        Returns:
            The absolute path of the file written.
        """

        # ----- helpers -----
        def _slug(s: str) -> str:
            s = (s or "").strip()
            s = re.sub(r"[^\w\-\.]+", "_", s)
            return s or "unnamed"

        def _norm_iri(s: str | None) -> str | None:
            if not s:
                return None
            s = s.strip().strip("<>")
            if s.startswith(("http://", "https://")):
                return s
            return None  # not an absolute IRI

        def _expand_curie_or_mint(pred: str) -> URIRef:
            """Expand a predicate that might be absolute IRI, CURIE, or plain string."""
            p_abs = _norm_iri(pred)
            if p_abs:
                return URIRef(p_abs)
            p = (pred or "").strip().strip("<>")
            # Expand CURIE if prefixes provided
            if ":" in p and prefixes:
                pref, local = p.split(":", 1)
                if pref in prefixes:
                    return URIRef(prefixes[pref].rstrip("#/") + "#" + local)
            # Mint under PROP ns
            return PROP[_slug(p)]

        def _concept_uri(c: Concept) -> URIRef:
            if c.iri:
                abs_ = _norm_iri(c.iri)
                if abs_:
                    return URIRef(abs_)
                # Treat c.iri as local name if it's not absolute
                return ENT[_slug(c.iri)]
            # Prefer a clean slug from name; fall back to nodeid
            local = _slug(c.name) if c.name else _slug(c.nodeid)
            return ENT[local]

        def _is_is_a(pred: str) -> bool:
            return is_hierarchy_predicate(pred)

        # ----- graph + namespaces -----
        g = Graph()

        if base_iri is None:
            base_iri = (f"http://example.org/{self.name}#", "ont")

        if not (base_iri[0].endswith("#") or base_iri[0].endswith("/")):
            base_iri_exp = base_iri[0] + "#"
        else:
            base_iri_exp = base_iri[0]

        ENT = Namespace(base_iri_exp)  # entity (classes) namespace
        PROP = Namespace(
            base_iri_exp + "prop/"
        )  # properties namespace (only used if we need to mint)
        ONT_IRI = URIRef(
            base_iri_exp.rstrip("#/")
        )  # ontology IRI (no trailing separator)
        OBO_IN_OWL = Namespace("http://www.geneontology.org/formats/oboInOwl#")

        # Standard bindings for nice output
        g.bind("rdf", RDF)
        g.bind("rdfs", RDFS)
        g.bind("owl", OWL)
        g.bind("xsd", XSD)
        g.bind(base_iri[1], ENT)
        g.bind("prop", PROP)
        g.bind("oboInOwl", OBO_IN_OWL)
        if prefixes:
            for pref, ns in prefixes.items():
                try:
                    g.bind(pref, Namespace(ns))
                except (TypeError, ValueError):
                    continue

        # ----- ontology header -----
        g.add((ONT_IRI, RDF.type, OWL.Ontology))
        g.add((ONT_IRI, RDFS.label, Literal(self.name)))
        g.add((ONT_IRI, OWL.versionInfo, Literal(self.version)))

        # ----- classes -----
        uri_cache: dict[Concept, URIRef] = {}
        for c in self.nodes:
            cu = _concept_uri(c)
            uri_cache[c] = cu
            g.add((cu, RDF.type, OWL.Class))
            if include_labels and c.name:
                for label in c.ground_set["labels"]:
                    g.add((cu, RDFS.label, Literal(label)))
                for alt_label in c.ground_set["alt_labels"]:
                    g.add((cu, SKOS.altLabel, Literal(alt_label)))
                for synonym in c.ground_set["related_synonyms"]:
                    g.add((cu, OBO_IN_OWL.hasRelatedSynonym, Literal(synonym)))
                for synonym in c.ground_set["exact_synonyms"]:
                    g.add((cu, OBO_IN_OWL.hasExactSynonym, Literal(synonym)))
                for definition in c.ground_set["definitions"]:
                    g.add((cu, SKOS.definition, Literal(definition)))

        # ----- edges -----
        for e in self.edges:
            s = uri_cache.get(e.src) or _concept_uri(e.src)
            o = uri_cache.get(e.tgt) or _concept_uri(e.tgt)

            if _is_is_a(e.pred):
                g.add((s, RDFS.subClassOf, o))
            else:
                p = _expand_curie_or_mint(e.pred)
                g.add((s, p, o))
                if declare_properties:
                    g.add((p, RDF.type, OWL.ObjectProperty))

        # ----- write to disk -----
        fmt = format.lower()
        if fmt in ("turtle", "ttl"):
            fmt = "turtle"
            ext = "ttl"
        elif fmt in ("xml", "rdfxml", "owl"):
            fmt = "xml"
            ext = "owl"
        else:
            raise ValueError(
                "format must be one of: 'turtle'/'ttl', 'xml'/'rdfxml'/'owl'"
            )

        if dirpath is None:
            dirpath = os.path.join("data", "ontologies")

        if filename is None:
            filename = f"ontology-{self.version}-{self.name}.{ext}"

        os.makedirs(dirpath, exist_ok=True)
        filepath = os.path.join(dirpath, filename)

        g.serialize(destination=filepath, format=fmt, encoding="utf-8")
        return os.path.abspath(filepath)

    def to_pickle(
        self,
        dirpath: str = "data/.cache/cross_domain/",  # cross_domain by default
        create_id: bool | None = False,
    ) -> str:
        """save ontology to .cache directory"""
        os.makedirs(dirpath, exist_ok=True)
        if create_id:
            key = f"{self.version}-{self.name}-{name_from_uuid(uuid.uuid4(), 'slug32')}"
        else:
            key = f"{self.version}-{self.name}"
        with open(os.path.join(dirpath, f"ontology-{key}.pkl"), "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        return key

    @staticmethod
    def load_pickle(key: str, dirpath: str = "data/.cache/cross_domain/") -> Ontology:
        """fast load from .cache directory"""
        with open(os.path.join(dirpath, f"ontology-{key}.pkl"), "rb") as f:
            return legacy_pickle.load(f)

    @staticmethod
    def union_ontologies(onts: list[Ontology]) -> Ontology:
        """
        Merge multiple Ontology objects:
        1) Combine all edges via triples -> build_ontology_from_triples
        2) Merge node metadata (iri + ground_set lists) for matching keys
        """
        if not onts:
            return Ontology()

        # 1) Gather all triples from edges
        triples: list[tuple[str, str, str]] = []
        for o in onts:
            for e in o.edges:
                triples.append((e.src.name, e.pred, e.tgt.name))

        # Build the merged structure (nodes/edges/parents/children/out_rels/in_rels)
        merged = Ontology().build_ontology_from_triples(triples)

        # 2) Merge node metadata
        def _merge_list(dst: list[str], src: list[str]):
            seen = set(dst)
            for s in src or []:
                s = (s or "").strip()
                if s and s not in seen:
                    dst.append(s)
                    seen.add(s)

        for o in onts:
            # Use the ontology's node_map keys so lookups are consistent with how edges were created
            for key, orig in o.node_map.items():
                # Ensure this key exists in merged (get_node creates it if missing)
                mn = merged.get_node(key)

                # Prefer to keep an existing IRI; otherwise take orig's IRI
                if orig.iri and not mn.iri:
                    mn.iri = orig.iri

                # Merge ground_set fields (keep your schema keys)
                # If any key is missing in either side, default to []
                for k in set(mn.ground_set.keys()) | set(orig.ground_set.keys()):
                    if k not in mn.ground_set:
                        mn.ground_set[k] = []
                    _merge_list(mn.ground_set[k], orig.ground_set.get(k, []))

        # Name/version
        merged.name = "union-" + "-".join(
            o.name for o in onts if getattr(o, "name", None)
        )
        merged.version = max(
            (getattr(o, "version", "0.1.0") for o in onts), default="0.1.0"
        )
        return merged
