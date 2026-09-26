"""Cosine similarity support for concept-graph nodes."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from agentoi.algorithms.graph import ConceptGraph

logger = logging.getLogger(__name__)


class SimilarityPlugin(ABC):
    """Abstract interface for graph similarity plugins."""

    @abstractmethod
    def compute_pairwise_similarity(
        self, graph: ConceptGraph, node_ids: list[str] | None = None
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def compute_similarity_to_query(
        self,
        graph: ConceptGraph,
        query_embedding: torch.Tensor,
        node_ids: list[str] | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class CosineSimilarityPlugin(SimilarityPlugin):
    """Compute cosine similarity over node embeddings."""

    def __init__(
        self,
        alpha: float = 0.5,
        normalize: bool = True,
        *,
        index_backend: str = "torch-exact",
        index_options: dict[str, Any] | None = None,
    ):
        self.alpha = alpha
        self.normalize = normalize
        self.index_backend = index_backend
        self.index_options = dict(index_options or {})

    def _get_node_embeddings(
        self, graph: ConceptGraph, node_ids: list[str] | None = None
    ) -> tuple[torch.Tensor, list[str]]:
        requested_ids = list(graph.nodes) if node_ids is None else node_ids
        embeddings = []
        valid_node_ids = []
        for node_id in requested_ids:
            if node_id not in graph.nodes:
                continue
            node = graph.nodes[node_id]
            if getattr(node, "embedding", None) is None:
                try:
                    node.compute_embedding(alpha=self.alpha)
                except Exception as error:
                    logger.warning(
                        "Could not compute embedding for node %s: %s", node_id, error
                    )
                    continue
            embeddings.append(node.embedding)
            valid_node_ids.append(node_id)
        if not embeddings:
            raise ValueError("No valid embeddings found for the specified nodes")
        return torch.stack(embeddings), valid_node_ids

    def compute_pairwise_similarity(
        self, graph: ConceptGraph, node_ids: list[str] | None = None
    ) -> torch.Tensor:
        embeddings, _ = self._get_node_embeddings(graph, node_ids)
        if self.normalize:
            embeddings = embeddings / (
                embeddings.norm(p=2, dim=1, keepdim=True) + 1e-9
            )
        return torch.mm(embeddings, embeddings.T)

    def compute_similarity_to_query(
        self,
        graph: ConceptGraph,
        query_embedding: torch.Tensor,
        node_ids: list[str] | None = None,
    ) -> torch.Tensor:
        embeddings, _ = self._get_node_embeddings(graph, node_ids)
        if self.normalize:
            query_embedding = query_embedding / (query_embedding.norm(p=2) + 1e-9)
            embeddings = embeddings / (
                embeddings.norm(p=2, dim=1, keepdim=True) + 1e-9
            )
        return torch.mv(embeddings, query_embedding)

    def get_top_similar_pairs(
        self,
        graph: ConceptGraph,
        k: int = 10,
        node_ids: list[str] | None = None,
        exclude_self: bool = True,
    ) -> list[tuple[str, str, float]]:
        embeddings, valid_node_ids = self._get_node_embeddings(graph, node_ids)
        if self.normalize:
            embeddings = embeddings / (
                embeddings.norm(p=2, dim=1, keepdim=True) + 1e-9
            )
        similarity_matrix = torch.mm(embeddings, embeddings.T)
        diagonal = 1 if exclude_self else 0
        mask = torch.triu(
            torch.ones_like(similarity_matrix), diagonal=diagonal
        ).bool()
        values = similarity_matrix[mask]
        if not len(values) or k <= 0:
            return []
        indices = torch.nonzero(mask, as_tuple=False)
        top_values, top_indices = torch.topk(values, k=min(k, len(values)))
        return [
            (
                valid_node_ids[indices[index][0].item()],
                valid_node_ids[indices[index][1].item()],
                top_values[position].item(),
            )
            for position, index in enumerate(top_indices)
        ]

    def get_most_similar_nodes(
        self,
        graph: ConceptGraph,
        query_embedding: torch.Tensor,
        k: int = 10,
        node_ids: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        embeddings, valid_node_ids = self._get_node_embeddings(graph, node_ids)
        if self.normalize:
            from agentoi.embeddings import create_vector_index

            index = create_vector_index(
                self.index_backend,
                **{
                    **self.index_options,
                    "dimension": int(embeddings.shape[1]),
                },
            )
            index.add(valid_node_ids, embeddings)
            return [
                (str(result.id), result.score)
                for result in index.search(query_embedding, k=min(k, len(valid_node_ids)))
            ]

        scores = torch.mv(embeddings, query_embedding)
        values, indices = torch.topk(scores, k=min(k, len(scores)))
        return [
            (valid_node_ids[index.item()], values[position].item())
            for position, index in enumerate(indices)
        ]


def compute_cosine_similarity_matrix(
    graph: ConceptGraph,
    alpha: float = 0.5,
    node_ids: list[str] | None = None,
    normalize: bool = True,
) -> torch.Tensor:
    return CosineSimilarityPlugin(alpha=alpha, normalize=normalize).compute_pairwise_similarity(
        graph, node_ids
    )


def compute_similarity_to_query(
    graph: ConceptGraph,
    query_embedding: torch.Tensor,
    alpha: float = 0.5,
    node_ids: list[str] | None = None,
    normalize: bool = True,
) -> torch.Tensor:
    return CosineSimilarityPlugin(alpha=alpha, normalize=normalize).compute_similarity_to_query(
        graph, query_embedding, node_ids
    )
