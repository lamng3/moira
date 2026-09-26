from __future__ import annotations

import os
import pickle
import uuid
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .builder import OntologyStructureBuilder
from .concept_graph import ConceptGraph
from .equivalence import DSU, EquivalentClass, EquivalentClassRelation
from .ontology import Ontology
from .predicates import (
    canonicalize_predicate,
    is_equivalence_predicate,
    is_hierarchy_predicate,
)

if TYPE_CHECKING:
    from agentoi.embeddings import BaseVectorIndex, TextEmbedding


@dataclass
class ConceptHyperedge:
    """Directed hyperedge over EquivalentClass nodes."""

    tail: list[EquivalentClass]
    head: list[EquivalentClass]
    predicate: str
    score: float = 1.0
    eid: str = field(default_factory=lambda: f"he_{uuid.uuid4().hex[:8]}")
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert self.tail and self.head, "Hyperedge requires non-empty tail and head."

        def _uniq(eqs: list[EquivalentClass]) -> list[EquivalentClass]:
            seen, out = set(), []
            for ec in eqs:
                if ec.id not in seen:
                    seen.add(ec.id)
                    out.append(ec)
            return out

        self.tail = _uniq(self.tail)
        self.head = _uniq(self.head)

    def describe(self) -> str:
        def _labels(eqs: list[EquivalentClass]) -> str:
            return "; ".join(ec.members(return_label=True) for ec in eqs)

        return f"[{self.predicate}|{self.score:.2f}] {{ {_labels(self.tail)} }} -> {{ {_labels(self.head)} }}"

    def __repr__(self) -> str:
        return f"ConceptHyperedge(eid={self.eid}, pred={self.predicate!r}, score={self.score}, tail={[t.id for t in self.tail]}, head={[h.id for h in self.head]})"


class ConceptHypergraph:
    """
    Hypergraph over EquivalentClass nodes with directed hyperedges.
    - build_from_ontology(): DSU equivalence collapse, then group into hyperedges
    - project_pairwise(): star-expands hyperedges for compatibility with GraphEmbedding / Node2Vec
    - compute_all_embeddings(): text + projected-graph embeddings, then fuse
    - nearest_nodes(): cosine NN over fused EC embeddings
    """

    def __init__(
        self,
        nodes: list[EquivalentClass] | None = None,
        hyperedges: list[ConceptHyperedge] | None = None,
    ):
        self.nodes: dict[str, EquivalentClass] = {n.id: n for n in (nodes or [])}
        self.hyperedges: list[ConceptHyperedge] = []
        self.out_incidence: dict[str, list[ConceptHyperedge]] = defaultdict(list)
        self.in_incidence: dict[str, list[ConceptHyperedge]] = defaultdict(list)
        self.parents: dict[str, list[EquivalentClass]] = defaultdict(list)  # projected
        self.children: dict[str, list[EquivalentClass]] = defaultdict(list)  # projected
        for he in hyperedges or []:
            self.add_hyperedge(he)

    # ---------- Build ----------

    def build_from_ontology(self, ont: Ontology) -> ConceptHypergraph:
        """Collapse equivalences, create ECs, then group non-equivalence edges into hyperedges."""
        if ont is None or not getattr(ont, "nodes", None):
            return self

        # reset
        self.nodes.clear()
        self.hyperedges.clear()
        self.out_incidence.clear()
        self.in_incidence.clear()
        self.parents.clear()
        self.children.clear()

        # ECs
        nodes, concept_to_eq = OntologyStructureBuilder.equivalence_classes(
            ont,
            dsu_factory=DSU,
            equivalent_class_factory=EquivalentClass,
        )
        for equivalent_class in nodes.values():
            self.add_node(equivalent_class)

        # group (src_eq, predicate) -> {tgt_eq}
        grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
        for e in ont.edges:
            if is_equivalence_predicate(e.pred):
                continue
            src_eq = concept_to_eq.get(e.src.nodeid)
            tgt_eq = concept_to_eq.get(e.tgt.nodeid)
            if not src_eq or not tgt_eq or src_eq is tgt_eq:
                continue
            pred = canonicalize_predicate(e.pred)
            grouped[(src_eq.id, pred)].add(tgt_eq.id)

        for (src_id, pred), tgt_ids in grouped.items():
            src = self.nodes[src_id]
            tgts = [self.nodes[tid] for tid in sorted(tgt_ids)]
            w = 2.0 if is_hierarchy_predicate(pred) else 1.0
            self.add_hyperedge(
                ConceptHyperedge(tail=[src], head=tgts, predicate=pred, score=w)
            )

        self.compute_ranks()
        return self

    # ---------- Mutators ----------

    def add_node(self, node: EquivalentClass) -> bool:
        if node is None:
            return False
        self.nodes[node.id] = node
        return True

    def add_hyperedge(self, he: ConceptHyperedge) -> bool:
        self.hyperedges.append(he)
        for n in he.tail + he.head:
            self.nodes.setdefault(n.id, n)
        for t in he.tail:
            self.out_incidence[t.id].append(he)
        for h in he.head:
            self.in_incidence[h.id].append(he)
        # maintain projected adjacency
        for t in he.tail:
            for h in he.head:
                if h not in self.children[t.id]:
                    self.children[t.id].append(h)
                if t not in self.parents[h.id]:
                    self.parents[h.id].append(t)
                if h not in t.children:
                    t.children.append(h)
                if t not in h.parents:
                    h.parents.append(t)
        return True

    # ---------- Projection ----------

    def project_pairwise(self, normalize_by_head: bool = True) -> ConceptGraph:
        """Expand each hyperedge into all tail×head pairs; optionally divide score by |head|."""
        nodes = list(self.nodes.values())
        cg = ConceptGraph(nodes=nodes, edges=[])

        for he in self.hyperedges:
            denom = float(len(he.head)) if (normalize_by_head and he.head) else 1.0
            w = he.score / denom
            for t in he.tail:
                for h in he.head:
                    rel = EquivalentClassRelation(src=t, tgt=h, relation="no", score=w)
                    rel.predicate = he.predicate  # stash predicate
                    cg.add_edge(rel)
        return cg

    # ---------- Embeddings ----------

    def compute_all_embeddings(
        self,
        alpha: float = 0.5,
        *,
        text_model: TextEmbedding | None = None,
        graph_params: dict[str, Any] | None = None,
        projection_normalize_by_head: bool = True,
        node2vec_model_path: str | None = None,
    ) -> ConceptHypergraph:
        """Text embeddings on ECs + Node2Vec on projection; then fuse."""
        from agentoi.embeddings import GraphEmbedding, TextEmbedding, fuse_embeddings

        txt = text_model or TextEmbedding()
        # text
        for ec in self.nodes.values():
            ec.text_embedding = txt.compute_embedding(ec)

        # projected graph + node2vec
        cg = self.project_pairwise(normalize_by_head=projection_normalize_by_head)
        ge = GraphEmbedding(
            cg,
            dimensions=txt.embed_dim,
            save_path=node2vec_model_path,
            **(graph_params or {}),
        )

        # fuse
        for ec in self.nodes.values():
            g = ge.embs[ec.id]
            t = ec.text_embedding
            ec.graph_embedding = g
            ec.embedding = fuse_embeddings(g, t, alpha)
        return self

    # ---------- NN over EC embeddings ----------

    def nearest_nodes(
        self,
        terms: str | list[str],
        k: int = 10,
        *,
        text_model: TextEmbedding | None = None,
        agg: str = "max",
        weights: Sequence[float] | None = None,
        return_out: bool = True,
        iri_prefix: str | Sequence[str] | None = None,
        exclusive: bool = False,
        index: BaseVectorIndex | None = None,
        index_backend: str = "torch-exact",
        index_options: dict[str, Any] | None = None,
    ):
        if not self.nodes:
            return []

        from agentoi.embeddings import MultiQuerySearchService, TextEmbedding

        return MultiQuerySearchService(
            index=index,
            backend=index_backend,
            backend_options=index_options,
        ).search(
            list(self.nodes.values()),
            terms,
            text_model=text_model or TextEmbedding(),
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
            top = [node for node, _ in self.nearest_nodes(terms, k=k)]
        else:
            top = [
                self.nodes[node_id]
                for node_id in selected_ids
                if node_id in self.nodes
            ]
        top_ids = {node.id for node in top}
        hyperedges = [
            edge
            for edge in self.hyperedges
            if all(node.id in top_ids for node in edge.tail + edge.head)
        ][:max_edges]

        lines = [f"QUERY: {query_text}", "RELEVANT CONCEPT CLUSTERS:"]
        for position, node in enumerate(top, 1):
            lines.append(f"{position}. {node.members(return_label=True)}")
        if hyperedges:
            lines.append("KEY RELATIONS (within top-k):")
            lines.extend(f"- {edge.describe()}" for edge in hyperedges)
        lines.append(
            "TASK: Using the above concept clusters and relations as contextual guidance, "
            "answer the user query precisely. Prefer concepts and relations that appear in "
            "the RELEVANT CONCEPT CLUSTERS and KEY RELATIONS. Cite specific terms."
        )
        return "\n".join(lines)

    # ---------- Ranks, cycles, utils ----------

    def compute_ranks(self) -> dict[str, int]:
        """Bottom-up ranks using only hierarchical hyperedges (score >= 2)."""
        children_isa: dict[str, list[EquivalentClass]] = defaultdict(list)
        for he in self.hyperedges:
            if getattr(he, "score", 0.0) >= 2.0:
                for t in he.tail:
                    for h in he.head:
                        children_isa[t.id].append(h)

        ranks: dict[str, int] = {}
        state: dict[str, int] = {}

        def dfs(n: EquivalentClass) -> int:
            st = state.get(n.id, 0)
            if st == 2:
                return ranks[n.id]
            if st == 1:
                return 0
            state[n.id] = 1
            kids = children_isa.get(n.id, [])
            r = 0 if not kids else 1 + max(dfs(k) for k in kids)
            state[n.id] = 2
            ranks[n.id] = r
            n.rank = r
            return r

        for n in self.nodes.values():
            if state.get(n.id, 0) == 0:
                dfs(n)
        return ranks

    def has_cycle(self) -> bool:
        """Cycle check via pairwise projection and union-find in the ConceptGraph."""
        cg = self.project_pairwise()
        return cg.has_cycle()

    @staticmethod
    def load_pickle(
        dirpath: str = "data/.cache/concept_hypergraph",
        filename: str = "concept_hypergraph.pkl",
    ) -> ConceptHypergraph:
        with open(os.path.join(dirpath, filename), "rb") as f:
            return pickle.load(f)

    def to_pickle(
        self,
        dirpath: str = "data/.cache/concept_hypergraph",
        filename: str = "concept_hypergraph.pkl",
    ) -> None:
        os.makedirs(dirpath, exist_ok=True)
        with open(os.path.join(dirpath, filename), "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def __repr__(self) -> str:
        return f"ConceptHypergraph(num_nodes={len(self.nodes)}, num_hyperedges={len(self.hyperedges)})"

    def __str__(self) -> str:
        lines = [repr(self)]
        for he in self.hyperedges:
            lines.append(f"  {he!r}")
        return "\n".join(lines)
