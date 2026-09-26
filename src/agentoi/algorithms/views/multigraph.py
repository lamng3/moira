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

from agentoi.algorithms.graph import ConceptGraph


class MultigraphView:
    """
    allow multiple parallel edges and loops.
    edges allow more than one edge between same two vertices, and self-loops.

    adjacency maps each source node id to a list of outgoing edge ids.
    reverse_adjacency maps each target node id to a list of incoming edge ids.
    """

    def __init__(
        self,
        graph: ConceptGraph,
        multiedges: Iterable[tuple[str, str, str, dict[str, Any]]] | None = None,
        *,
        carry_attrs: bool = True,
    ):
        # nodes / attrs
        self.nodes: set[str] = set()
        self.node_attrs: dict[str, dict[str, Any]] = defaultdict(dict)

        # edges / attrs
        self.edges: list[tuple[str, str, str, dict]] = []
        self.edge_attrs: dict[str, dict[str, Any]] = {}
        self.edge_index: dict[str, tuple[str, str]] = {}

        # fast adjacency
        self.adjacency: dict[str, list[str]] = defaultdict(list)
        self.reverse_adjacency: dict[str, list[str]] = defaultdict(list)

        # create from ConceptGraph
        if graph and graph.nodes:
            # register nodes (EC ids)
            for eq in graph.nodes.values():
                self._register_node(eq.id)
                if carry_attrs:
                    self.node_attrs[eq.id]["name"] = eq.members(return_label=False)
                    self.node_attrs[eq.id]["label"] = eq.members(return_label=True)
                    # ranks if computed
                    if getattr(eq, "rank", None) is not None:
                        self.node_attrs[eq.id]["rank"] = eq.rank
                    # parents/children by EC id
                    self.node_attrs[eq.id]["parents"] = [
                        p.id for p in getattr(eq, "parents", [])
                    ]
                    self.node_attrs[eq.id]["children"] = [
                        c.id for c in getattr(eq, "children", [])
                    ]

                    # embeddings (if present)
                    if getattr(eq, "embedding", None) is not None:
                        self.node_attrs[eq.id].setdefault("embedding", eq.embedding)
                    if getattr(eq, "text_embedding", None) is not None:
                        self.node_attrs[eq.id].setdefault(
                            "text_embedding", eq.text_embedding
                        )
                    if getattr(eq, "graph_embedding", None) is not None:
                        self.node_attrs[eq.id].setdefault(
                            "graph_embedding", eq.graph_embedding
                        )

            # import each ConceptGraph edge as a multiedge with a stable id
            for idx, e in enumerate(graph.edges):
                src_id, tgt_id = e.src.id, e.tgt.id
                edge_id = f"cg_{idx:08d}_{src_id}__{tgt_id}"
                meta = {
                    "origin": "concept_graph",
                    "relation": getattr(
                        e, "relation", None
                    ),  # "yes"/"no" in your EC relation schema
                    "score": float(getattr(e, "score", 0.0)),
                    # convenience (human-readable members)
                    "src_members": e.src.members(),
                    "tgt_members": e.tgt.members(),
                }
                self.add_edge(edge_id, src_id, tgt_id, meta)

        # register any additional multiedges provided by caller
        if multiedges:
            for edge_id, src_id, tgt_id, metadata in multiedges:
                self.add_edge(edge_id, src_id, tgt_id, metadata)

    def _register_node(self, node_id: str) -> None:
        if node_id not in self.nodes:
            self.nodes.add(node_id)
            # ensure adjacency dicts exist
            self.adjacency.setdefault(node_id, [])
            self.reverse_adjacency.setdefault(node_id, [])
            self.node_attrs.setdefault(node_id, {})

    def add_node(self, node_id: str, attrs: dict | None = None) -> None:
        self._register_node(node_id)
        if attrs:
            self.node_attrs[node_id].update(attrs or {})

    def add_edge(
        self,
        edge_id: str,
        src_id: str,
        tgt_id: str,
        metadata: dict | None = None,
    ) -> None:
        """add a parallel edge or loop to the multigraph"""
        if src_id not in self.nodes:
            self._register_node(src_id)
        if tgt_id not in self.nodes:
            self._register_node(tgt_id)

        meta = metadata or {}
        self.edges.append((edge_id, src_id, tgt_id, meta))
        self.edge_attrs[edge_id] = meta
        self.edge_index[edge_id] = (src_id, tgt_id)

        self.adjacency[src_id].append(edge_id)
        self.reverse_adjacency[tgt_id].append(edge_id)

    def remove_edge(self, edge_id: str) -> bool:
        """Remove all edges with the given id (usually unique)."""
        found = False
        for e in list(self.edges):
            if e[0] == edge_id:
                _, src_id, tgt_id, _ = e
                self.edges.remove(e)
                # update adjacency
                if edge_id in self.adjacency.get(src_id, []):
                    self.adjacency[src_id].remove(edge_id)
                if edge_id in self.reverse_adjacency.get(tgt_id, []):
                    self.reverse_adjacency[tgt_id].remove(edge_id)
                # attrs & index
                self.edge_attrs.pop(edge_id, None)
                self.edge_index.pop(edge_id, None)
                found = True
        return found

    def edges_between(self, src_id: str, tgt_id: str) -> list[str]:
        """list edge ids connecting src_id to tgt_id"""
        return [eid for eid, s, t, _ in self.edges if s == src_id and t == tgt_id]

    def incident_edges(self, node_id: str) -> list[str]:
        """list edge ids incident to node_id (including loops)"""
        out = list(self.adjacency.get(node_id, []))
        inc = list(self.reverse_adjacency.get(node_id, []))
        # build a new dictinary whose keys are items in the concatenated list in order
        return list(dict.fromkeys(out + inc))

    def num_nodes(self) -> int:
        return len(self.nodes)

    def num_edges(self) -> int:
        return len(self.edges)

    def out_degree(self, node_id: str) -> int:
        return len(self.adjacency.get(node_id, []))

    def in_degree(self, node_id: str) -> int:
        return len(self.reverse_adjacency.get(node_id, []))

    # ---------- NetworkX interop & viz ----------
    def to_nx(self) -> nx.MultiDiGraph:
        """
        Export to a NetworkX MultiDiGraph.
        Node attributes and edge attributes are preserved.
        """
        G = nx.MultiDiGraph()
        # nodes with attrs
        for nid in self.nodes:
            G.add_node(nid, **(self.node_attrs.get(nid, {})))
        # edges with attrs (keep edge_id as key 'id')
        for edge_id, src, tgt, meta in self.edges:
            # MultiDiGraph supports parallel edges; store our ID in edge data
            G.add_edge(src, tgt, key=edge_id, id=edge_id, **(meta or {}))
        return G

    def draw(
        self,
        *,
        figsize: tuple[int, int] = (10, 8),
        layout: Literal[
            "spring", "fr", "kk", "spectral", "random", "planar"
        ] = "spring",
        with_labels: bool = False,
        seed: int = 42,
        node_size_base: float = 60.0,
        node_size_exp: float = 0.9,
        edge_width_min: float = 0.75,
        edge_width_max: float = 3.0,
        alpha_nodes: float = 0.9,
        alpha_edges: float = 0.5,
        weight_attr: str = "score",  # which edge attr to use as width weight
        title: str | None = None,
        filepath: str = "data/.cache/multigraph/multigraph_view.png",
    ) -> nx.MultiDiGraph:
        """
        Draw the multigraph by collapsing parallel edges into effective weights
        for visualization (NetworkX path-based drawing doesn't show parallel arcs by default).
        """
        Gm = self.to_nx()
        if Gm.number_of_nodes() == 0:
            print("Multigraph is empty.")
            return Gm

        # Build a simple DiGraph view with summed weights for widths
        G = nx.DiGraph()
        for u, v, k, d in Gm.edges(keys=True, data=True):
            w = float(d.get(weight_attr, 1.0))
            if G.has_edge(u, v):
                G[u][v]["weight"] += w
            else:
                G.add_edge(u, v, weight=w)

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
        nx.draw_networkx_edges(
            G, pos, width=widths.tolist(), alpha=alpha_edges, arrows=True
        )
        if with_labels:
            # prefer 'label' attr if present
            labels = {n: self.node_attrs.get(n, {}).get("label", n) for n in G.nodes()}
            # label can be a list; shorten
            labels = {
                n: (", ".join(labels[n]) if isinstance(labels[n], list) else labels[n])
                for n in labels
            }
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

        if title:
            plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath)
        return Gm

    @staticmethod
    def load_pickle(
        dirpath: str = "data/.cache/multigraph", filename: str = "multigraph.pkl"
    ) -> MultigraphView:
        """fast load from .cache directory"""
        with open(os.path.join(dirpath, filename), "rb") as f:
            return pickle.load(f)

    def to_pickle(
        self, dirpath: str = "data/.cache/multigraph", filename: str = "multigraph.pkl"
    ) -> None:
        """Save MultigraphView to .cache directory."""
        os.makedirs(dirpath, exist_ok=True)
        with open(os.path.join(dirpath, filename), "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def __repr__(self) -> str:
        return f"MultigraphView(#nodes={len(self.nodes)}, #edges={len(self.edges)})"

    def __str__(self) -> str:
        lines = [repr(self)]
        for eid, src, tgt, meta in self.edges:
            rel = meta.get("relation", None)
            score = meta.get("score", None)
            lines.append(f"  {eid}: {src} -> {tgt}  (relation={rel}, score={score})")
        return "\n".join(lines)
