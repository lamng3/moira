"""Public API for similarity plugins, engines, and result types."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "SimilarityMethod": ("types", "SimilarityMethod"),
    "SimilarityResult": ("types", "SimilarityResult"),
    "ComparisonResult": ("types", "ComparisonResult"),
    "CrossGraphPair": ("types", "CrossGraphPair"),
    "NodePairSimilarityEngine": ("engine", "NodePairSimilarityEngine"),
    "SimilarityPlugin": ("cosine", "SimilarityPlugin"),
    "CosineSimilarityPlugin": ("cosine", "CosineSimilarityPlugin"),
    "compute_cosine_similarity_matrix": ("cosine", "compute_cosine_similarity_matrix"),
    "compute_similarity_to_query": ("cosine", "compute_similarity_to_query"),
    "SimilarityComparisonSystem": ("comparison", "SimilarityComparisonSystem"),
    "create_similarity_comparison_system": (
        "comparison",
        "create_similarity_comparison_system",
    ),
    "compare_similarity_methods": ("comparison", "compare_similarity_methods"),
    "CrossGraphSimilarity": ("cross_graph", "CrossGraphSimilarity"),
    "find_top_k_cross_graph_pairs": ("cross_graph", "find_top_k_cross_graph_pairs"),
    "compare_cross_graph_methods": ("cross_graph", "compare_cross_graph_methods"),
    "analyze_cross_graph_similarity": ("cross_graph", "analyze_cross_graph_similarity"),
    "NormalizedGoogleDistance": ("ngd", "NormalizedGoogleDistance"),
    "compute_google_similarity_matrix": (
        "ngd",
        "compute_google_similarity_matrix",
    ),
    "compute_google_similarity_to_query": (
        "ngd",
        "compute_google_similarity_to_query",
    ),
    "EmbeddingAwareSimilaritySystem": (
        "integration",
        "EmbeddingAwareSimilaritySystem",
    ),
    "create_embedding_aware_system": (
        "integration",
        "create_embedding_aware_system",
    ),
    "analyze_alignment_similarity": (
        "integration",
        "analyze_alignment_similarity",
    ),
    "analyze_similarity_methods_with_views": (
        "integration",
        "analyze_similarity_methods_with_views",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value
