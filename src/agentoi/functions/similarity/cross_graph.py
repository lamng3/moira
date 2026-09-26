"""Similarity scoring for node pairs from two concept graphs."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from .cosine import CosineSimilarityPlugin
from .engine import NodePairSimilarityEngine
from .ngd import NormalizedGoogleDistance
from .types import CrossGraphPair, SimilarityMethod

if TYPE_CHECKING:
    from agentoi.algorithms.graph import ConceptGraph, EquivalentClass

logger = logging.getLogger(__name__)


class CrossGraphSimilarity:
    """Find similar nodes across two concept graphs."""

    def __init__(
        self,
        cosine_plugin: CosineSimilarityPlugin | None = None,
        ngd: NormalizedGoogleDistance | None = None,
        hybrid_weight: float = 0.5,
        cache_results: bool = True,
    ):
        self.engine = NodePairSimilarityEngine(
            cosine_plugin=cosine_plugin,
            ngd=ngd,
            hybrid_weight=hybrid_weight,
        )
        self.cosine_plugin = self.engine.cosine_plugin
        self.ngd = self.engine.ngd
        self.hybrid_weight = self.engine.hybrid_weight
        self.cache_results = cache_results
        self._similarity_cache: dict[tuple[Any, ...], float] = {}

    def _get_cache_key(
        self,
        node_a: EquivalentClass,
        node_b: EquivalentClass,
        method: str,
    ) -> tuple[Any, ...]:
        """Return an identity-aware, graph-direction-preserving cache key."""
        return (
            self.engine.node_signature(node_a),
            self.engine.node_signature(node_b),
            method,
        )

    def compute_cross_graph_similarity(
        self,
        node_a: EquivalentClass,
        node_b: EquivalentClass,
        method: SimilarityMethod,
    ) -> float:
        cache_key = self._get_cache_key(node_a, node_b, method.value)
        if self.cache_results and cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]

        similarity = self.engine.score(node_a, node_b, method)
        if not isinstance(similarity, float):
            raise TypeError("Cross-graph scoring requires a scalar score")
        if self.cache_results:
            self._similarity_cache[cache_key] = similarity
        return similarity

    def find_top_k_cross_graph_pairs(
        self,
        graph_a: ConceptGraph,
        graph_b: ConceptGraph,
        k: int = 10,
        method: SimilarityMethod = SimilarityMethod.HYBRID,
        similarity_threshold: float = 0.0,
        node_ids_a: list[str] | None = None,
        node_ids_b: list[str] | None = None,
    ) -> list[CrossGraphPair]:
        ids_a = list(graph_a.nodes) if node_ids_a is None else node_ids_a
        ids_b = list(graph_b.nodes) if node_ids_b is None else node_ids_b
        all_pairs = []
        for node_id_a in ids_a:
            if node_id_a not in graph_a.nodes:
                continue
            node_a = graph_a.nodes[node_id_a]
            for node_id_b in ids_b:
                if node_id_b not in graph_b.nodes:
                    continue
                node_b = graph_b.nodes[node_id_b]
                similarity = self.compute_cross_graph_similarity(
                    node_a, node_b, method
                )
                if similarity >= similarity_threshold:
                    all_pairs.append(
                        CrossGraphPair(
                            node_a_id=node_id_a,
                            node_b_id=node_id_b,
                            node_a=node_a,
                            node_b=node_b,
                            similarity_score=similarity,
                            method=method.value,
                            metadata={
                                "graph_a_size": len(graph_a.nodes),
                                "graph_b_size": len(graph_b.nodes),
                                "total_pairs_considered": len(ids_a) * len(ids_b),
                            },
                        )
                    )
        all_pairs.sort(key=lambda pair: pair.similarity_score, reverse=True)
        return all_pairs[:k]

    def compare_cross_graph_methods(
        self,
        graph_a: ConceptGraph,
        graph_b: ConceptGraph,
        k: int = 10,
        similarity_threshold: float = 0.0,
        node_ids_a: list[str] | None = None,
        node_ids_b: list[str] | None = None,
    ) -> dict[str, list[CrossGraphPair]]:
        results = {}
        for method in SimilarityMethod:
            try:
                results[method.value] = self.find_top_k_cross_graph_pairs(
                    graph_a,
                    graph_b,
                    k=k,
                    method=method,
                    similarity_threshold=similarity_threshold,
                    node_ids_a=node_ids_a,
                    node_ids_b=node_ids_b,
                )
            except Exception as error:
                logger.exception(
                    "Error computing %s cross-graph similarity: %s",
                    method.value,
                    error,
                )
                results[method.value] = []
        return results

    def analyze_cross_graph_similarity(
        self,
        graph_a: ConceptGraph,
        graph_b: ConceptGraph,
        method: SimilarityMethod = SimilarityMethod.HYBRID,
        node_ids_a: list[str] | None = None,
        node_ids_b: list[str] | None = None,
    ) -> dict[str, Any]:
        count_a = len(graph_a.nodes) if node_ids_a is None else len(node_ids_a)
        count_b = len(graph_b.nodes) if node_ids_b is None else len(node_ids_b)
        all_pairs = self.find_top_k_cross_graph_pairs(
            graph_a,
            graph_b,
            k=count_a * count_b,
            method=method,
            node_ids_a=node_ids_a,
            node_ids_b=node_ids_b,
        )
        if not all_pairs:
            return {"error": "No pairs found for analysis"}
        scores = [pair.similarity_score for pair in all_pairs]
        return {
            "total_pairs": len(all_pairs),
            "method_used": method.value,
            "statistics": {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores),
                "median": np.median(scores),
            },
            "score_distribution": {
                "high_similarity": sum(score >= 0.8 for score in scores),
                "medium_similarity": sum(0.4 <= score < 0.8 for score in scores),
                "low_similarity": sum(score < 0.4 for score in scores),
            },
            "top_pairs": [
                {
                    "node_a_id": pair.node_a_id,
                    "node_b_id": pair.node_b_id,
                    "node_a_members": pair.node_a.members(return_label=True),
                    "node_b_members": pair.node_b.members(return_label=True),
                    "similarity_score": pair.similarity_score,
                }
                for pair in all_pairs[:10]
            ],
        }


def find_top_k_cross_graph_pairs(
    graph_a: ConceptGraph,
    graph_b: ConceptGraph,
    k: int = 10,
    method: SimilarityMethod = SimilarityMethod.HYBRID,
    similarity_threshold: float = 0.0,
    **kwargs: Any,
) -> list[CrossGraphPair]:
    return CrossGraphSimilarity(**kwargs).find_top_k_cross_graph_pairs(
        graph_a,
        graph_b,
        k=k,
        method=method,
        similarity_threshold=similarity_threshold,
    )


def compare_cross_graph_methods(
    graph_a: ConceptGraph,
    graph_b: ConceptGraph,
    k: int = 10,
    similarity_threshold: float = 0.0,
    **kwargs: Any,
) -> dict[str, list[CrossGraphPair]]:
    return CrossGraphSimilarity(**kwargs).compare_cross_graph_methods(
        graph_a, graph_b, k=k, similarity_threshold=similarity_threshold
    )


def analyze_cross_graph_similarity(
    graph_a: ConceptGraph,
    graph_b: ConceptGraph,
    method: SimilarityMethod = SimilarityMethod.HYBRID,
    **kwargs: Any,
) -> dict[str, Any]:
    return CrossGraphSimilarity(**kwargs).analyze_cross_graph_similarity(
        graph_a, graph_b, method=method
    )
