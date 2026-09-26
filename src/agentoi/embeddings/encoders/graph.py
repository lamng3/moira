from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx
import torch
from node2vec import Node2Vec

if TYPE_CHECKING:
    from agentoi.algorithms.graph import ConceptGraph, ConceptHypergraph, EquivalentClass

_N2V_WORKERS = max(1, min(8, (os.cpu_count() or 4) - 1))
DEFAULT_GRAPH_DIMENSION = 128


class GraphEmbedding:
    """Node2Vec encoder for concept graphs and projected hypergraphs.

    The standalone default is 128 dimensions. Fusion pipelines pass the text
    encoder's dimension explicitly so graph and text vectors remain compatible.
    """

    def __init__(
        self,
        cg_or_chg: ConceptGraph | ConceptHypergraph | None = None,
        dimensions: int = DEFAULT_GRAPH_DIMENSION,
        save_path: str | None = None,
        **n2v_kwargs,
    ):
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions
        self.n2v_kwargs = {"quiet": True, **n2v_kwargs}
        self.save_path = save_path
        self.G: nx.DiGraph = nx.DiGraph()
        self.embs: dict[str, torch.Tensor] = {}

        if cg_or_chg is not None:
            self.from_graphlike(cg_or_chg)
            self.embs = self.learn_node2vec()

    def compute_embedding(self, node: EquivalentClass) -> torch.Tensor:
        """Retrieve a node vector, retraining if the graph has changed."""
        if not self.embs:
            raise ValueError(
                "No embeddings found: call from_graphlike() then learn_node2vec()."
            )
        if node.id not in self.embs:
            self.embs = self.learn_node2vec()
        embedding = self.embs[node.id]
        setattr(node, "graph_embedding", embedding)
        return embedding

    def from_concept_graph(self, cg: ConceptGraph) -> nx.DiGraph:
        self.G.clear()
        for node in cg.nodes.values():
            self.G.add_node(node.id)
        for edge in cg.edges:
            self.G.add_edge(
                edge.src.id,
                edge.tgt.id,
                weight=getattr(edge, "score", 1.0),
            )
        return self.G

    def from_concept_hypergraph(self, chg: ConceptHypergraph) -> nx.DiGraph:
        return self.from_concept_graph(chg.project_pairwise())

    def from_graphlike(
        self, obj: ConceptGraph | ConceptHypergraph
    ) -> nx.DiGraph:
        if hasattr(obj, "project_pairwise"):
            return self.from_concept_hypergraph(obj)  # type: ignore[arg-type]
        return self.from_concept_graph(obj)  # type: ignore[arg-type]

    def learn_node2vec(
        self,
        walk_length: int = 10,
        num_walks: int = 50,
        p: float = 1.0,
        q: float = 1.0,
        workers: int | None = None,
        window: int = 5,
        epochs: int = 1,
        save_path: str | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.G.number_of_nodes() == 0:
            raise ValueError("Empty graph")

        path = Path(save_path or self.save_path) if save_path or self.save_path else None
        if path and path.exists():
            from gensim.models import Word2Vec

            model = Word2Vec.load(str(path))
            return {
                node: torch.tensor(model.wv.get_vector(node), dtype=torch.float32)
                for node in self.G.nodes()
                if node in model.wv
            }

        node2vec = Node2Vec(
            self.G,
            dimensions=self.dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            p=p,
            q=q,
            workers=workers if workers is not None else _N2V_WORKERS,
            weight_key="weight",
            **self.n2v_kwargs,
        )
        model = node2vec.fit(window=window, min_count=1, epochs=epochs)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(path))
        return {
            node: torch.tensor(model.wv.get_vector(node), dtype=torch.float32)
            for node in self.G.nodes()
        }
