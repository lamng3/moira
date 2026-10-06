"""Shared node-pair similarity scoring engine."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from .cosine import CosineSimilarityPlugin
from .ngd import NormalizedGoogleDistance
from .types import SimilarityMethod

if TYPE_CHECKING:
    from moira.algorithms.graph import EquivalentClass

logger = logging.getLogger(__name__)


class NodePairSimilarityEngine:
    """Compute cosine, NGD-derived, and hybrid scores for two nodes."""

    def __init__(
        self,
        cosine_plugin: CosineSimilarityPlugin | None = None,
        ngd: NormalizedGoogleDistance | None = None,
        hybrid_weight: float = 0.5,
    ):
        if not 0.0 <= hybrid_weight <= 1.0:
            raise ValueError("hybrid_weight must be between 0.0 and 1.0")
        self.cosine_plugin = cosine_plugin or CosineSimilarityPlugin()
        self.ngd = ngd or NormalizedGoogleDistance()
        self.hybrid_weight = hybrid_weight

    def node_signature(self, node: EquivalentClass) -> tuple[int, int, int, str]:
        """Return cache state that tracks vector and representative-term changes."""
        embedding = getattr(node, "embedding", None)
        version = getattr(embedding, "_version", 0)
        if not isinstance(version, int):
            version = 0
        try:
            term = self.ngd.representative_term(node)
        except (AttributeError, TypeError):
            term = str(getattr(node, "id", ""))
        return (
            id(node),
            id(embedding),
            version,
            term,
        )

    def cosine(self, node_1: EquivalentClass, node_2: EquivalentClass) -> float:
        """Return a cosine score clamped to the package's [0, 1] range."""
        for node in (node_1, node_2):
            if getattr(node, "embedding", None) is None:
                node.compute_embedding(alpha=self.cosine_plugin.alpha)

        embedding_1 = node_1.embedding
        embedding_2 = node_2.embedding
        if self.cosine_plugin.normalize:
            embedding_1 = embedding_1 / (embedding_1.norm() + 1e-9)
            embedding_2 = embedding_2 / (embedding_2.norm() + 1e-9)
        score = torch.dot(embedding_1, embedding_2).item()
        return max(0.0, min(1.0, float(score)))

    def google_index(
        self, node_1: EquivalentClass, node_2: EquivalentClass
    ) -> float:
        """Return the NGD-derived similarity score for two nodes."""
        try:
            term_1 = self.ngd.representative_term(node_1)
            term_2 = self.ngd.representative_term(node_2)
            score = self.ngd.compute_similarity(term_1, term_2)
            return max(0.0, min(1.0, float(score)))
        except Exception as error:
            logger.warning(
                "Error computing Google similarity between %s and %s: %s",
                getattr(node_1, "id", ""),
                getattr(node_2, "id", ""),
                error,
            )
            return 0.0

    def all_scores(
        self, node_1: EquivalentClass, node_2: EquivalentClass
    ) -> tuple[float, float, float]:
        """Return cosine, Google-index, and hybrid scores."""
        cosine_score = self.cosine(node_1, node_2)
        google_score = self.google_index(node_1, node_2)
        hybrid_score = (
            (1.0 - self.hybrid_weight) * cosine_score
            + self.hybrid_weight * google_score
        )
        return cosine_score, google_score, hybrid_score

    def score(
        self,
        node_1: EquivalentClass,
        node_2: EquivalentClass,
        method: SimilarityMethod,
        *,
        return_all: bool = False,
    ) -> float | tuple[float, float, float]:
        """Score a node pair with one method."""
        if method is SimilarityMethod.COSINE:
            return self.cosine(node_1, node_2)
        if method is SimilarityMethod.GOOGLE_INDEX:
            return self.google_index(node_1, node_2)
        if method is SimilarityMethod.HYBRID:
            scores = self.all_scores(node_1, node_2)
            return scores if return_all else scores[2]
        raise ValueError(f"Unknown similarity method: {method}")
