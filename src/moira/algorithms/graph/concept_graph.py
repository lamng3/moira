from __future__ import annotations

import os
import pickle
from collections import defaultdict
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .builder import OntologyStructureBuilder
from .concept import Concept
from .equivalence import DSU, EquivalentClass, EquivalentClassRelation
from .ontology import Ontology
from .predicates import is_equivalence_predicate, is_hierarchy_predicate

if TYPE_CHECKING:
    import torch

    from moira.embeddings import BaseVectorIndex, TextEmbedding


class ConceptGraph:
    """a DAG whose nodes are equivalent classes"""

    def __init__(
        self, nodes: list[EquivalentClass], edges: list[EquivalentClassRelation]
    ):
        self.nodes: dict[str, EquivalentClass] = {
            n.id: n for n in nodes
        }  # fast node lookup
        self.edges: list[EquivalentClassRelation] = []
        self.parents: dict[str, list[EquivalentClass]] = defaultdict(list)
        self.children: dict[str, list[EquivalentClass]] = defaultdict(list)
        # add initial edges
        for e in edges:
            self.add_edge(e)

    def build_from_ontology(self, ont: Ontology):
        """
        Build this ConceptGraph from a single Ontology:
        1) collapse owl:equivalentClass / owl:sameAs / skos:exactMatch into DSU groups
        2) create EquivalentClass nodes
        3) add directed edges between classes (is-a edges weighted higher)
        """
        if ont is None or not ont.nodes:
            return self

        # --- create EquivalentClass nodes and index ---
        self.nodes.clear()
        self.edges.clear()
        self.parents.clear()
        self.children.clear()
        nodes, concept_to_eq = OntologyStructureBuilder.equivalence_classes(
            ont,
            dsu_factory=DSU,
            equivalent_class_factory=EquivalentClass,
        )
        for equivalent_class in nodes.values():
            self.add_node(equivalent_class)

        # --- add edges between classes ---
        seen: set[tuple[str, str]] = set()  # (src_eq_id, tgt_eq_id)
        for e in ont.edges:
            if is_equivalence_predicate(e.pred):
                continue  # skip internal equivalence edges
            src_eq = concept_to_eq.get(e.src.nodeid)
            tgt_eq = concept_to_eq.get(e.tgt.nodeid)
            if src_eq is None or tgt_eq is None or src_eq is tgt_eq:
                continue
            key = (src_eq.id, tgt_eq.id)
            if key in seen:
                continue
            seen.add(key)

            w = 2.0 if is_hierarchy_predicate(e.pred) else 1.0
            rel = EquivalentClassRelation(src_eq, tgt_eq, relation="no", score=w)
            self.add_edge(rel)

        # record rank (optional)
        self.compute_ranks()
        return self

    def add_node(self, node: EquivalentClass) -> bool:
        """add an equivalent class to concept graph, return status"""
        if node is not None:
            self.nodes[node.id] = node
            return True
        return False

    def add_edge(self, edge: EquivalentClassRelation) -> bool:
        self.edges.append(edge)
        self.parents[edge.tgt.id].append(edge.src)
        self.children[edge.src.id].append(edge.tgt)
        # ensure both nodes exist
        self.nodes.setdefault(edge.src.id, edge.src)
        self.nodes.setdefault(edge.tgt.id, edge.tgt)
        # also reflect on the EC objects themselves
        if edge.src not in edge.tgt.parents:
            edge.tgt.parents.append(edge.src)
        if edge.tgt not in edge.src.children:
            edge.src.children.append(edge.tgt)
        return True

    def compute_all_embeddings(
        self,
        alpha: float = 0.5,
        *,
        text_model: TextEmbedding | None = None,
        graph_params: dict[str, Any] | None = None,
        node2vec_model_path: str | None = None,
        progress: Any | None = None,
        control: Any | None = None,
    ):
        """
        Creates node2vec embeddings on this graph, text embeddings for each EC,
        and stores:
            node.graph_embedding, node.text_embedding, node.embedding
        """
        from moira.embeddings import GraphEmbedding, TextEmbedding, fuse_embeddings

        if progress is not None:
            progress.stage(
                "Loading the text embedding model. The first run may download it."
            )
        txt = text_model or TextEmbedding()
        self.text_encoder = txt
        if progress is not None:
            progress.stage("Learning graph structure embeddings.")
        ge = GraphEmbedding(
            self,
            dimensions=txt.embed_dim,
            save_path=node2vec_model_path,
            **(graph_params or {}),
        )
        # ge.embs is already populated in GraphEmbedding.__init__ when cg is provided

        from moira.progress import computing_status

        nodes = list(self.nodes.values())
        total = len(nodes)
        for index, node in enumerate(nodes, start=1):
            if control is not None:
                control.raise_if_cancelled()
            if progress is not None:
                progress.tick(computing_status(index, total))
            # text
            node.text_embedding = txt.compute_embedding(node)
            # graph
            if node.id not in ge.embs:
                # if graph changed, retrain once
                ge.embs = ge.learn_node2vec()
            node.graph_embedding = ge.embs[node.id]
            # fuse
            node.embedding = fuse_embeddings(
                node.graph_embedding,
                node.text_embedding,
                alpha,
            )

        if progress is not None and total:
            progress.stage(f"Embedded {total} concepts.")

        return self  # fluent

    def nearest_nodes(
        self,
        terms: str | list[str],
        k: int = 10,
        *,
        text_model: TextEmbedding | None = None,
        agg: str = "max",  # 'max' | 'mean' | 'sum' | 'weighted'
        weights: Sequence[float] | None = None,
        return_out: bool = True,
        iri_prefix: str | Sequence[str] | None = None,  # NEW
        exclusive: bool = False,  # NEW: require all members to match prefix
        index: BaseVectorIndex | None = None,
        index_backend: str = "torch-exact",
        index_options: dict[str, Any] | None = None,
    ) -> list[tuple[EquivalentClass, float]] | tuple[torch.Tensor, torch.Tensor]:
        """
        Return top-k nodes by cosine similarity to a set of query terms.

        If `iri_prefix` is provided, only consider ECs that belong to that ontology:
        - By default (exclusive=False), an EC is kept if *any* member Concept has
            iri/name starting with one of the prefixes.
        - If exclusive=True, an EC is kept only if *all* member Concepts match.
        """
        if not self.nodes:
            return []

        from moira.embeddings import MultiQuerySearchService, TextEmbedding

        encoder = text_model or getattr(self, "text_encoder", None)
        if encoder is None:
            encoder = TextEmbedding()
            self.text_encoder = encoder
        return MultiQuerySearchService(
            index=index,
            backend=index_backend,
            backend_options=index_options,
        ).search(
            list(self.nodes.values()),
            terms,
            text_model=encoder,
            k=k,
            agg=agg,
            weights=weights,
            return_out=return_out,
            iri_prefix=iri_prefix,
            exclusive=exclusive,
        )

    def make_prompt_for_query(
        self,
        query_text: str,
        terms: list[str],
        *,
        k: int = 10,
        max_edges: int = 40,
        selected_ids: Sequence[str] | None = None,
    ) -> str:
        if selected_ids is None:
            top = [n for n, _ in self.nearest_nodes(terms, k=k)]
        else:
            top = [self.nodes[node_id] for node_id in selected_ids if node_id in self.nodes]
        top_ids = {n.id for n in top}
        # induced edges among top nodes
        edges = [e for e in self.edges if e.src.id in top_ids and e.tgt.id in top_ids][
            :max_edges
        ]

        # brief node summaries
        lines = [f"QUERY: {query_text}", "CONCEPTS:"]
        for n in top:
            lines.append(f"- {n.members(return_label=True)}")

        if edges:
            lines.append("RELATIONS:")
            for e in edges:
                lines.append(
                    f"- {e.src.members(return_label=True)}  --({e.score:.1f})-->  {e.tgt.members(return_label=True)}"
                )

        lines.append(
            "Answer the QUERY in plain sentences using the names above. "
            "Do not refer to this list, its headings, or these instructions."
        )
        return "\n".join(lines)

    def parents_of(self, node: EquivalentClass) -> list[EquivalentClass]:
        return list(self.parents.get(node.id, []))

    def children_of(self, node: EquivalentClass) -> list[EquivalentClass]:
        return list(self.children.get(node.id, []))

    def compute_ranks(self) -> dict[str, int]:
        """
        Compute bottom-up ranks using only hierarchical (is-a) edges and
        be robust to cycles (treat a detected back-edge as a leaf boundary).
        """
        # Build adjacency from edges, but only for is-a edges (score==2.0 in build)
        children_isa: dict[str, list[EquivalentClass]] = defaultdict(list)
        for e in self.edges:
            if (
                getattr(e, "score", 0.0) >= 2.0
            ):  # 2.0 was set for is-a edges in build_from_ontology
                children_isa[e.src.id].append(e.tgt)

        ranks: dict[str, int] = {}
        state: dict[str, int] = {}  # 0=unvisited, 1=visiting, 2=done

        def dfs(node: EquivalentClass) -> int:
            st = state.get(node.id, 0)
            if st == 2:
                return ranks[node.id]
            if st == 1:
                # cycle detected -> break the recursion by treating this edge as length 0
                return 0

            state[node.id] = 1
            kids = children_isa.get(node.id, [])
            r = 0 if not kids else 1 + max(dfs(k) for k in kids)
            state[node.id] = 2
            ranks[node.id] = r
            node.rank = r
            return r

        for n in self.nodes.values():
            if state.get(n.id, 0) == 0:
                dfs(n)

        return ranks

    @staticmethod
    def load_pickle(
        dirpath: str = "data/.cache/concept_graph", filename: str = "concept_graph.pkl"
    ) -> ConceptGraph:
        """fast load from .cache directory"""
        with open(os.path.join(dirpath, filename), "rb") as f:
            return pickle.load(f)

    def to_pickle(
        self,
        dirpath: str = "data/.cache/concept_graph",
        filename: str = "concept_graph.pkl",
    ) -> None:
        """save ConceptGraph to .cache directory"""
        os.makedirs(dirpath, exist_ok=True)
        with open(os.path.join(dirpath, filename), "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def __getstate__(self):
        """Custom pickle state to avoid circular references"""
        # Create a serializable state without circular references
        state = {}

        # Store basic attributes
        state["nodes"] = {}
        for node_id, node in self.nodes.items():
            # Create a serializable version of the node
            # Convert Concept objects to serializable dictionaries
            equiv_concepts_serialized = []
            for concept in node.equiv_concepts:
                concept_dict = {
                    "name": concept.name,
                    "nodeid": concept.nodeid,
                    "iri": concept.iri,
                    "rank": concept.rank,
                    "ground_set": concept.ground_set,
                    "parents": [p.name for p in concept.parents if hasattr(p, "name")],
                    "children": [
                        c.name for c in concept.children if hasattr(c, "name")
                    ],
                    "out_rels": {
                        k: [c.name for c in v if hasattr(c, "name")]
                        for k, v in concept.out_rels.items()
                    },
                    "in_rels": {
                        k: [c.name for c in v if hasattr(c, "name")]
                        for k, v in concept.in_rels.items()
                    },
                }
                equiv_concepts_serialized.append(concept_dict)

            node_state = {
                "id": node.id,
                "equiv_concepts": equiv_concepts_serialized,
                "rank": node.rank,
                "text_embedding": node.text_embedding,
                "graph_embedding": node.graph_embedding,
                "embedding": node.embedding,
                "parents": [
                    parent.id for parent in node.parents if hasattr(parent, "id")
                ],
                "children": [
                    child.id for child in node.children if hasattr(child, "id")
                ],
            }
            state["nodes"][node_id] = node_state

        # Store edges with ID references
        state["edges"] = []
        for edge in self.edges:
            edge_state = {
                "src": edge.src.id if hasattr(edge.src, "id") else None,
                "tgt": edge.tgt.id if hasattr(edge.tgt, "id") else None,
                "relation": edge.relation,
                "score": edge.score,
            }
            state["edges"].append(edge_state)

        # Store parent/child dictionaries with ID references
        state["parents"] = {
            k: [n.id for n in v if hasattr(n, "id")] for k, v in self.parents.items()
        }
        state["children"] = {
            k: [n.id for n in v if hasattr(n, "id")] for k, v in self.children.items()
        }

        return state

    def __setstate__(self, state):
        """Restore state after unpickling (fast path with ID-based wiring)."""
        # 1) Recreate EC nodes and Concepts (no links yet)
        self.nodes = {}
        for node_id, node_state in state["nodes"].items():
            equiv_concepts = []
            for cd in node_state["equiv_concepts"]:
                c = Concept(
                    name=cd.get("name"),
                    nodeid=cd.get("nodeid"),
                    iri=cd.get("iri"),
                )
                c.rank = cd.get("rank")
                c.ground_set = cd.get("ground_set") or {
                    "labels": [],
                    "alt_labels": [],
                    "related_synonyms": [],
                    "exact_synonyms": [],
                    "definitions": [],
                }
                # Stash raw link data; we'll resolve with maps
                c._parent_keys = cd.get("parents", [])  # nodeids preferred
                c._child_keys = cd.get("children", [])
                c._out_rels_d = cd.get("out_rels", {})
                c._in_rels_d = cd.get("in_rels", {})
                # For legacy pickles that stored names, keep a fallback
                c._parent_names = (
                    cd.get("parents_names", []) if "parents_names" in cd else []
                )
                c._child_names = (
                    cd.get("children_names", []) if "children_names" in cd else []
                )
                equiv_concepts.append(c)

            node = EquivalentClass(equiv_concepts=equiv_concepts)
            node.id = node_state["id"]
            node.rank = node_state["rank"]
            node.text_embedding = node_state["text_embedding"]
            node.graph_embedding = node_state["graph_embedding"]
            node.embedding = node_state["embedding"]
            node._parent_ids = node_state.get("parents", [])
            node._child_ids = node_state.get("children", [])
            self.nodes[node_id] = node

        # 2) Build fast lookups once
        #    - EC by id
        ec_by_id = {n.id: n for n in self.nodes.values()}
        #    - Concept by nodeid (primary) and by name (fallback for legacy)
        concept_by_nodeid = {}
        concept_by_name = {}
        for ec in self.nodes.values():
            for c in ec.equiv_concepts:
                if c.nodeid:
                    concept_by_nodeid[c.nodeid] = c
                if c.name:
                    # Store first encounter only; names can collide but we avoid O(N^2) scanning
                    concept_by_name.setdefault(c.name, c)

        # Helper to resolve a list of concept keys that might be nodeids or names
        def _resolve_concepts(keys, names_fallback=None):
            out = []
            for k in keys or []:
                c = concept_by_nodeid.get(k)
                if c is None and names_fallback is not None:
                    c = concept_by_name.get(k)
                if c is not None:
                    out.append(c)
            return out

        # 3) Rebuild EC parents/children in one pass
        self.parents = defaultdict(list)
        self.children = defaultdict(list)
        for node_id, ec in self.nodes.items():
            if hasattr(ec, "_parent_ids"):
                ec.parents = [ec_by_id[i] for i in ec._parent_ids if i in ec_by_id]
                delattr(ec, "_parent_ids")
            if hasattr(ec, "_child_ids"):
                ec.children = [ec_by_id[i] for i in ec._child_ids if i in ec_by_id]
                delattr(ec, "_child_ids")

        # 4) Rebuild edges in a single pass (no placeholders)
        self.edges = []
        for e in state["edges"]:
            src = ec_by_id.get(e["src"])
            tgt = ec_by_id.get(e["tgt"])
            if src is None or tgt is None:
                continue
            rel = EquivalentClassRelation(
                src=src, tgt=tgt, relation=e["relation"], score=e["score"]
            )
            self.edges.append(rel)

        # 5) Refill parent/child dicts from state (IDs -> objects)
        for k, ids in state.get("parents", {}).items():
            self.parents[k] = [ec_by_id[i] for i in ids if i in ec_by_id]
        for k, ids in state.get("children", {}).items():
            self.children[k] = [ec_by_id[i] for i in ids if i in ec_by_id]

        # 6) Rebuild Concept-level links using the precomputed maps
        #    This is now O(total links) instead of O(num_concepts^2)
        for ec in self.nodes.values():
            for c in ec.equiv_concepts:
                # parents/children: prefer nodeids; fall back to legacy names if needed
                if hasattr(c, "_parent_keys"):
                    c.parents = _resolve_concepts(
                        c._parent_keys, names_fallback=None
                    ) or _resolve_concepts(
                        getattr(c, "_parent_names", []), names_fallback=True
                    )
                    delattr(c, "_parent_keys")
                    if hasattr(c, "_parent_names"):
                        delattr(c, "_parent_names")
                if hasattr(c, "_child_keys"):
                    c.children = _resolve_concepts(
                        c._child_keys, names_fallback=None
                    ) or _resolve_concepts(
                        getattr(c, "_child_names", []), names_fallback=True
                    )
                    delattr(c, "_child_keys")
                    if hasattr(c, "_child_names"):
                        delattr(c, "_child_names")

                # out_rels / in_rels
                from collections import defaultdict as _dd

                c.out_rels = _dd(list)
                c.in_rels = _dd(list)
                if hasattr(c, "_out_rels_d"):
                    for r, lst in (c._out_rels_d or {}).items():
                        for key in lst:
                            t = concept_by_nodeid.get(key) or concept_by_name.get(key)
                            if t:
                                c.out_rels[r].append(t)
                    delattr(c, "_out_rels_d")
                if hasattr(c, "_in_rels_d"):
                    for r, lst in (c._in_rels_d or {}).items():
                        for key in lst:
                            t = concept_by_nodeid.get(key) or concept_by_name.get(key)
                            if t:
                                c.in_rels[r].append(t)
                    delattr(c, "_in_rels_d")

    def has_cycle(self) -> bool:
        """Return whether the directed concept graph contains a cycle."""
        state: dict[str, int] = {}

        def visit(node_id: str) -> bool:
            if state.get(node_id) == 1:
                return True
            if state.get(node_id) == 2:
                return False
            state[node_id] = 1
            if any(visit(child.id) for child in self.children.get(node_id, ())):
                return True
            state[node_id] = 2
            return False

        for node_id in self.nodes:
            if state.get(node_id, 0) == 0 and visit(node_id):
                return True
        return False

    def __repr__(self) -> str:
        return f"ConceptGraph(num_nodes={len(self.nodes)}, num_edges={len(self.edges)})"

    def __str__(self) -> str:
        lines = [repr(self)]
        for node in self.nodes.values():
            lines.append(
                f"  {node!r}  children -> {[c.id for c in self.children_of(node)]}     parents -> {[c.id for c in self.parents_of(node)]}"
            )
        return "\n".join(lines)
