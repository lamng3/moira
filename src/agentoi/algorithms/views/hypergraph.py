from __future__ import annotations

import math
import os
import pickle
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Literal

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import sparse

from agentoi.algorithms.graph import Concept, ConceptGraph


def _default_concept_id(c):
    return getattr(c, "nodeid", getattr(c, "name", str(id(c))))


class HypergraphView:
    """
    a concept can only belong to one equivalence class
    a hyperedge can contain multiple concepts coming from multiple equivalence classes
    """

    def __init__(
        self,
        graph: ConceptGraph,
        hyperedges: dict[str, set[Concept]] | None = None,
        *,
        concept_id_fn=None,
    ):
        self.concept_id_fn = concept_id_fn or _default_concept_id
        # nodes
        self.nodes: set[str] = set()
        self.node_index: dict[str, int] = {}  # for matrix rows
        self.node_attrs: dict[str, dict[str, Any]] = defaultdict(dict)

        # hyperedges
        self.hyperedges: dict[str, set[str]] = {}
        self.edge_index: dict[str, int] = {}  # for matrix cols
        self.edge_attrs: dict[str, dict[str, Any]] = defaultdict(dict)

        # incidence: vid -> set(eid)
        self.incidence: dict[str, set[str]] = defaultdict(set)
        self.edge_parents: dict[str, list[str]] = defaultdict(list)
        self.edge_children: dict[str, list[str]] = defaultdict(list)

        # register each equivalence class as a hyperedge by its id
        if graph and graph.nodes:
            for eq in graph.nodes.values():
                members = {self.concept_id_fn(c) for c in eq.equiv_concepts}
                self._register_hyperedge(eq.id, members)

                # carry over embedding as edge-level embedding
                if getattr(eq, "embedding", None) is not None:
                    self.edge_attrs[eq.id]["embedding"] = eq.embedding
                # name & label: a readable list of members
                self.edge_attrs[eq.id]["name"] = eq.members(return_label=False)
                self.edge_attrs[eq.id]["label"] = eq.members(return_label=True)

                # preserve EC hierarchy as attrs
                self.edge_attrs[eq.id]["parents"] = [
                    p.id for p in getattr(eq, "parents", [])
                ]
                self.edge_attrs[eq.id]["children"] = [
                    c.id for c in getattr(eq, "children", [])
                ]

                # also seed per-node attrs with text/graph/fused if available at EC-level
                for c in eq.equiv_concepts:
                    vid = self.concept_id_fn(c)
                    # you may map concept->EC fused embedding if you wish:
                    if getattr(eq, "embedding", None) is not None:
                        self.node_attrs[vid].setdefault("embedding", eq.embedding)
                    if getattr(eq, "text_embedding", None) is not None:
                        self.node_attrs[vid].setdefault(
                            "text_embedding", eq.text_embedding
                        )
                    if getattr(eq, "graph_embedding", None) is not None:
                        self.node_attrs[vid].setdefault(
                            "graph_embedding", eq.graph_embedding
                        )

        # Build edge hierarchy once; rebuilding it per EC creates duplicates.
        for edge_id, attrs in self.edge_attrs.items():
            for parent_id in attrs.get("parents", []):
                self.edge_parents[edge_id].append(parent_id)
                self.edge_children[parent_id].append(edge_id)

        # register any extra hyperedges
        if hyperedges:
            for eid, members in hyperedges.items():
                mem_ids = {self.concept_id_fn(c) for c in members}
                self._register_hyperedge(eid, mem_ids)

        # finalize indices
        self._reindex()

    def _register_hyperedge(self, edge_id: str, member_ids: set[str]) -> None:
        """add hyperedge and update incidence"""
        if not member_ids:
            return  # ignore empty
        for old_member_id in self.hyperedges.get(edge_id, set()):
            self.incidence[old_member_id].discard(edge_id)
        self.hyperedges[edge_id] = set(member_ids)
        for vid in member_ids:
            self.nodes.add(vid)
            self.incidence[vid].add(edge_id)

    def add_hyperedge(self, edge_id: str, members: Iterable[str]) -> None:
        """
        add a hyperedge with given id and member concepts
        overwrite if edge_id already exists
        """
        mem_ids = set(members)
        self._register_hyperedge(edge_id, mem_ids)
        self._reindex()

    def add_concepts_to_edge(self, edge_id: str, members: Iterable[str]) -> None:
        if edge_id not in self.hyperedges:
            self.hyperedges[edge_id] = set()
        for vid in members:
            self.nodes.add(vid)
            self.hyperedges[edge_id].add(vid)
            self.incidence[vid].add(edge_id)
        self._reindex()

    def remove_hyperedge(self, edge_id: str) -> bool:
        """
        return an entire hyperedge by its id
        update incidence of member nodes
        """
        members = self.hyperedges.pop(edge_id, None)
        if members is None:
            return False
        for c in members:
            self.incidence[c].discard(edge_id)
        self.edge_attrs.pop(edge_id, None)
        self._reindex()
        return True

    def ec_size_stats(self):
        sizes = [len(vids) for vids in self.hyperedges.values()]
        if not sizes:
            print("No hyperedges.")
            return
        singles = sum(1 for s in sizes if s == 1)
        print(
            f"Hyperedges: {len(sizes)}  |  min/max: {min(sizes)}/{max(sizes)}  |  mean: {sum(sizes) / len(sizes):.2f}"
        )
        print(f"Singleton hyperedges: {singles} ({singles / len(sizes):.1%})")

    def to_nx_two_section(
        self,
        *,
        min_weight: int = 1,
        top_k_per_node: int | None = None,
        keep_nodes: Iterable[str] | None = None,
    ) -> nx.Graph:
        """
        Build the 2-section (clique expansion) as a weighted simple graph.
        - min_weight: drop co-membership edges with weight < min_weight
        - top_k_per_node: keep only the strongest k incident edges per node (ties arbitrary)
        - keep_nodes: optional whitelist of node IDs to keep; drops edges touching others
        """
        co = self.two_section()
        G = nx.Graph()

        # optional node filter
        keep = set(keep_nodes) if keep_nodes is not None else None

        # first pass: add all edges above threshold (respect node filter if provided)
        for (a, b), w in co.items():
            if w < min_weight:
                continue
            if keep is not None and (a not in keep or b not in keep):
                continue
            G.add_edge(a, b, weight=float(w))

        if top_k_per_node is not None and top_k_per_node > 0:
            # prune per-node to top-k heaviest edges
            # collect best neighbors per node
            to_keep: set[tuple[str, str]] = set()
            for u in list(G.nodes()):
                nbrs = [(u, v, G[u][v]["weight"]) for v in G.neighbors(u)]
                nbrs.sort(key=lambda x: x[2], reverse=True)
                for _, v, _ in nbrs[:top_k_per_node]:
                    a, b = (u, v) if u < v else (v, u)
                    to_keep.add((a, b))
            # rebuild graph with kept edges only
            H = nx.Graph()
            H.add_nodes_from(G.nodes())
            for a, b in to_keep:
                H.add_edge(a, b, weight=G[a][b]["weight"])
            G = H

        return G

    def draw_two_section(
        self,
        *,
        min_weight: int = 1,
        top_k_per_node: int | None = None,
        keep_nodes: Iterable[str] | None = None,
        layout: Literal[
            "spring", "fr", "kk", "spectral", "random", "planar"
        ] = "spring",
        figsize: tuple[int, int] = (10, 8),
        with_labels: bool = False,
        seed: int = 42,
        node_size_base: float = 60.0,
        node_size_exp: float = 0.9,
        edge_width_min: float = 0.4,
        edge_width_max: float = 4.0,
        alpha_nodes: float = 0.9,
        alpha_edges: float = 0.5,
        title: str | None = None,
        filepath: str = "data/.cache/hypergraph/hypergraph_view.png",
    ) -> nx.Graph:
        """
        Build and draw the 2-section. Returns the NetworkX graph used for drawing.
        Node size scales ~ degree^node_size_exp; edge width scales with weight.

        For very large graphs, consider setting top_k_per_node or keep_nodes.
        """
        G = self.to_nx_two_section(
            min_weight=min_weight, top_k_per_node=top_k_per_node, keep_nodes=keep_nodes
        )

        if G.number_of_nodes() == 0:
            print("Two-section is empty after filtering.")
            return G

        # choose a layout
        if layout == "spring":
            pos = nx.spring_layout(G, seed=seed)
        elif layout == "fr":
            pos = nx.fruchterman_reingold_layout(G, seed=seed)
        elif layout == "kk":
            pos = nx.kamada_kawai_layout(G)
        elif layout == "spectral":
            pos = nx.spectral_layout(G)
        elif layout == "random":
            pos = nx.random_layout(G, seed=seed)
        elif layout == "planar":
            try:
                pos = nx.planar_layout(G)
            except nx.NetworkXException:
                pos = nx.spring_layout(G, seed=seed)
        else:
            pos = nx.spring_layout(G, seed=seed)

        # node sizes by degree
        deg = dict(G.degree())
        deg_vals = np.array([deg[n] for n in G.nodes()], dtype=float)
        node_sizes = node_size_base * np.power(np.maximum(1.0, deg_vals), node_size_exp)

        # edge widths by weight (min-max normalized)
        weights = np.array([G[u][v]["weight"] for u, v in G.edges()], dtype=float)
        if weights.size:
            wmin, wmax = float(weights.min()), float(weights.max())
            if math.isclose(wmin, wmax):
                widths = np.full_like(weights, (edge_width_min + edge_width_max) / 2.0)
            else:
                widths = edge_width_min + (weights - wmin) * (
                    edge_width_max - edge_width_min
                ) / (wmax - wmin)
        else:
            widths = np.array([])

        plt.figure(figsize=figsize)
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=alpha_nodes)
        nx.draw_networkx_edges(G, pos, width=widths.tolist(), alpha=alpha_edges)
        if with_labels:
            nx.draw_networkx_labels(G, pos, font_size=8)

        if title:
            plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(filepath)
        return G

    def draw_star_expansion(
        self,
        *,
        figsize: tuple[int, int] = (11, 8),
        with_labels: bool = False,
        seed: int = 42,
        vertex_node_size: float = 40.0,
        edge_node_size: float = 80.0,
        alpha_vertices: float = 0.9,
        alpha_edges: float = 0.7,
        min_size: int = 2,
        title: str | None = "Star expansion (bipartite V-E)",
        filepath: str = "data/.cache/hypergraph/hypergraph_view.png",
    ) -> nx.Graph:
        """
        Visualize the bipartite star expansion (preserves hypergraph structure):
        - left partition: concept vertices (V)
        - right partition: hyperedge nodes (E)
        """
        v_nodes, e_nodes, _ = self.star_expansion()
        if not v_nodes or not e_nodes:
            print("Star expansion is empty.")
            return nx.Graph()
        # filter edges by hyperedge size
        filtered_edges = []
        filtered_e_nodes = []
        for e in e_nodes:
            vids = self.hyperedges[e]
            if len(vids) >= min_size:
                for v in vids:
                    filtered_edges.append((v, e))
                filtered_e_nodes.append(e)
        if not filtered_edges:
            print(f"No hyperedges with size >= {min_size}")
            return nx.Graph()

        B = nx.Graph()
        B.add_nodes_from(v_nodes, bipartite=0, kind="node")
        B.add_nodes_from(filtered_e_nodes, bipartite=1, kind="hyperedge")
        B.add_edges_from(filtered_edges)

        # layout: project to 2D using spring; fix partitions on left/right
        rng = np.random.default_rng(seed)
        x_left = -1.0
        x_right = 1.0
        pos = {}
        # random vertical jitter for readability
        for i, v in enumerate(v_nodes):
            pos[v] = (
                x_left,
                (i / max(1, len(v_nodes) - 1)) * 2 - 1 + 0.01 * rng.standard_normal(),
            )
        for i, e in enumerate(filtered_e_nodes):
            pos[e] = (
                x_right,
                (i / max(1, len(filtered_e_nodes) - 1)) * 2
                - 1
                + 0.01 * rng.standard_normal(),
            )

        plt.figure(figsize=figsize)
        # vertices
        nx.draw_networkx_nodes(
            B,
            pos,
            nodelist=v_nodes,
            node_size=vertex_node_size,
            alpha=alpha_vertices,
        )
        # hyperedges
        nx.draw_networkx_nodes(
            B,
            pos,
            nodelist=filtered_e_nodes,
            node_shape="s",
            node_size=edge_node_size,
            alpha=alpha_edges,
        )
        nx.draw_networkx_edges(B, pos, alpha=0.3)
        if with_labels:
            nx.draw_networkx_labels(B, pos, font_size=7)

        if title:
            plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(filepath)
        return B

    def _reindex(self) -> None:
        self.node_index = {vid: i for i, vid in enumerate(sorted(self.nodes))}
        self.edge_index = {
            eid: j for j, eid in enumerate(sorted(self.hyperedges.keys()))
        }

    def hyperedges_of(self, concept: Concept) -> list[str]:
        """return list of hyperedges a given concept belongs to"""
        return list(self.incidence.get(concept.nodeid, []))

    def nodes_of(self, edge_id: str) -> set[str]:
        """Return concept IDs belonging to a hyperedge."""
        return set(self.hyperedges.get(edge_id, []))

    def degree(self, concept: Concept) -> int:
        """return the number of hyperedges the concept participates in"""
        return len(self.incidence.get(concept.nodeid, []))

    def num_nodes(self) -> int:
        return len(self.nodes)

    def num_hyperedges(self) -> int:
        return len(self.hyperedges)

    def parents_of_edge(self, edge_id: str) -> list[str]:
        return list(self.edge_parents.get(edge_id, []))

    def children_of_edge(self, edge_id: str) -> list[str]:
        return list(self.edge_children.get(edge_id, []))

    def incidence_matrix(self) -> sparse.coo_matrix:
        """|V| x |E| matrix with 1 if node∈edge else 0 (use weights if you add them)."""
        rows, cols, data = [], [], []
        for eid, vids in self.hyperedges.items():
            j = self.edge_index[eid]
            for vid in vids:
                i = self.node_index[vid]
                rows.append(i)
                cols.append(j)
                data.append(1.0)
        n, m = self.num_nodes(), self.num_hyperedges()
        return sparse.coo_matrix((data, (rows, cols)), shape=(n, m))

    def two_section(self) -> dict[tuple[str, str], int]:
        """
        2-section (clique expansion) on nodes.
        Returns dict of pair->weight (co-membership count).
        """
        co = defaultdict(int)
        for vids in self.hyperedges.values():
            vs = list(vids)
            L = len(vs)
            for a_idx in range(L):
                for b_idx in range(a_idx + 1, L):
                    a, b = vs[a_idx], vs[b_idx]
                    key = (a, b) if a < b else (b, a)
                    co[key] += 1
        return dict(co)

    def star_expansion(self) -> tuple[list[str], list[str], list[tuple[str, str]]]:
        """
        Bipartite representation (V nodes, E nodes, and edges between them).
        Useful for exact hypergraph algorithms/visualization.
        """
        v_nodes = sorted(self.nodes)
        e_nodes = sorted(self.hyperedges.keys())
        edges = []
        for eid, vids in self.hyperedges.items():
            for vid in vids:
                edges.append((vid, eid))
        return v_nodes, e_nodes, edges

    @staticmethod
    def load_pickle(
        dirpath: str = "data/.cache/hypergraph", filename: str = "hypergraph.pkl"
    ) -> HypergraphView:
        """fast load from .cache directory"""
        with open(os.path.join(dirpath, filename), "rb") as f:
            return pickle.load(f)

    def to_pickle(
        self, dirpath: str = "data/.cache/hypergraph", filename: str = "hypergraph.pkl"
    ):
        """save Hypergraph to .cache directory"""
        os.makedirs(dirpath, exist_ok=True)
        with open(os.path.join(dirpath, filename), "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def __repr__(self) -> str:
        return f"HypergraphView(|V|={self.num_nodes()}, |E|={self.num_hyperedges()})"

    def __str__(self) -> str:
        lines = [repr(self)]
        for eid, vids in sorted(self.hyperedges.items()):
            name = self.edge_attrs.get(eid, {}).get("name", "")
            label = self.edge_attrs.get(eid, {}).get("label", "")
            parents = self.parents_of_edge(eid)
            children = self.children_of_edge(eid)
            lines.append(
                f"  {eid} ({name} -- {label}): {sorted(vids)} -- parents -> {parents} -- children -> {children}"
            )
        return "\n".join(lines)
