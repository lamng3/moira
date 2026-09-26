"""Find related concepts across two graphs."""

import torch

from agentoi.algorithms.graph import Concept, ConceptGraph, EquivalentClass
from agentoi.functions.similarity import (
    SimilarityMethod,
    find_top_k_cross_graph_pairs,
)


def build_graph(prefix: str, labels: list[str], embeddings: list[list[float]]) -> ConceptGraph:
    """Build a small graph with predefined embeddings."""
    nodes = []
    for index, (label, embedding) in enumerate(zip(labels, embeddings)):
        concept = Concept(
            nodeid=f"{prefix}-{index}",
            name=label,
            ground_set={"labels": [label]},
        )
        node = EquivalentClass([concept])
        node.id = f"{prefix}-{index}"
        node.embedding = torch.tensor(embedding)
        nodes.append(node)
    return ConceptGraph(nodes=nodes, edges=[])


def main() -> None:
    envo = build_graph(
        "envo",
        ["ocean", "forest", "mountain"],
        [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]],
    )
    sweet = build_graph(
        "sweet",
        ["sea", "woodland", "peak"],
        [[0.95, 0.05], [0.05, 0.95], [0.65, 0.75]],
    )

    pairs = find_top_k_cross_graph_pairs(
        envo,
        sweet,
        k=3,
        method=SimilarityMethod.COSINE,
    )

    for pair in pairs:
        left = pair.node_a.members(return_label=True)
        right = pair.node_b.members(return_label=True)
        print(f"{left} ↔ {right}: {pair.similarity_score:.3f}")


if __name__ == "__main__":
    main()
