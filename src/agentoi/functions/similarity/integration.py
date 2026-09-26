"""Embedding-aware integration for the similarity engine and graph views."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from agentoi.retrieval.web import DatasetSearchProvider, WebSearchProvider

from .comparison import SimilarityComparisonSystem
from .cosine import CosineSimilarityPlugin
from .ngd import NormalizedGoogleDistance
from .types import ComparisonResult, SimilarityMethod

if TYPE_CHECKING:
    from agentoi.algorithms.graph import ConceptGraph
    from agentoi.algorithms.views import HypergraphView, MultigraphView


class EmbeddingAwareSimilaritySystem(SimilarityComparisonSystem):
    """Similarity comparison with graph embedding and view integration."""

    def __init__(
        self,
        concept_graph: "ConceptGraph",
        cosine_alpha: float = 0.5,
        google_base_frequency: int = 1000,
        hybrid_weight: float = 0.5,
        cache_results: bool = True,
        ensure_embeddings: bool = True,
        search_provider: WebSearchProvider | None = None,
        provider_seed: int | None = 0,
    ):
        if search_provider is None:
            search_provider = DatasetSearchProvider(
                concept_graph=concept_graph,
                base_frequency=google_base_frequency,
                seed=provider_seed,
            )
        cosine_plugin = CosineSimilarityPlugin(alpha=cosine_alpha)
        ngd = NormalizedGoogleDistance(search_provider=search_provider)
        super().__init__(
            cosine_plugin=cosine_plugin,
            ngd=ngd,
            hybrid_weight=hybrid_weight,
            cache_results=cache_results,
        )
        self.concept_graph = concept_graph
        self.ensure_embeddings = ensure_embeddings
        self.search_provider = search_provider

    def _ensure_embeddings_computed(
        self, node_ids: list[str] | None = None
    ) -> None:
        """Ensure embeddings are computed for specified nodes."""
        if not self.ensure_embeddings:
            return

        if node_ids is None:
            node_ids = list(self.concept_graph.nodes.keys())

        missing = any(
            node_id in self.concept_graph.nodes
            and any(
                getattr(self.concept_graph.nodes[node_id], attribute, None) is None
                for attribute in ("embedding", "text_embedding", "graph_embedding")
            )
            for node_id in node_ids
        )
        if missing:
            try:
                self.concept_graph.compute_all_embeddings(
                    alpha=self.cosine_plugin.alpha
                )
            except Exception as error:
                raise RuntimeError("Failed to compute graph embeddings") from error

    def compute_similarity(
        self,
        graph: "ConceptGraph",
        node_id_1: str,
        node_id_2: str,
        method: SimilarityMethod,
        return_all: bool = False,
    ) -> float | tuple[float, float, float]:
        """Override to ensure embeddings are computed."""
        self._ensure_embeddings_computed([node_id_1, node_id_2])
        return super().compute_similarity(graph, node_id_1, node_id_2, method, return_all)

    def compare_similarities(
        self,
        graph: "ConceptGraph",
        node_id_1: str,
        node_id_2: str,
    ) -> ComparisonResult:
        """Override to ensure embeddings are computed."""
        self._ensure_embeddings_computed([node_id_1, node_id_2])
        return super().compare_similarities(graph, node_id_1, node_id_2)

    def get_top_k_pairs(
        self,
        graph: "ConceptGraph",
        k: int = 10,
        method: SimilarityMethod = SimilarityMethod.COSINE,
        node_ids: list[str] | None = None,
        exclude_self: bool = True,
    ):
        """Override to ensure embeddings are computed."""
        self._ensure_embeddings_computed(node_ids)
        return super().get_top_k_pairs(graph, k, method, node_ids, exclude_self)

    def compare_top_k_pairs(
        self,
        graph: "ConceptGraph",
        k: int = 10,
        node_ids: list[str] | None = None,
        exclude_self: bool = True,
    ):
        """Override to ensure embeddings are computed."""
        self._ensure_embeddings_computed(node_ids)
        return super().compare_top_k_pairs(graph, k, node_ids, exclude_self)

    def integrate_with_views(
        self,
        graph: "ConceptGraph",
        hypergraph_view: "HypergraphView | None" = None,
        multigraph_view: "MultigraphView | None" = None,
    ) -> dict[str, Any]:
        """Enhanced integration with views."""
        results = super().integrate_with_views(graph, hypergraph_view, multigraph_view)

        get_vocabulary_stats = getattr(self.search_provider, "get_vocabulary_stats", None)
        if callable(get_vocabulary_stats):
            vocab_stats = get_vocabulary_stats()
            results["vocabulary"] = {
                "total_terms": vocab_stats.total_terms,
                "unique_terms": vocab_stats.unique_terms,
                "coverage": vocab_stats.vocabulary_coverage,
                "top_terms": sorted(
                    vocab_stats.term_frequencies.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:10],
            }

        if self.ensure_embeddings:
            results["embeddings"] = self._analyze_embeddings()
        return results

    def _analyze_embeddings(self) -> dict[str, Any]:
        """Analyze embedding characteristics."""
        stats: dict[str, Any] = {}
        for attribute, key in (
            ("embedding", "fused_embeddings"),
            ("text_embedding", "text_embeddings"),
            ("graph_embedding", "graph_embeddings"),
        ):
            values = [
                value.detach().cpu().numpy()
                for node in self.concept_graph.nodes.values()
                if (value := getattr(node, attribute, None)) is not None
            ]
            if not values:
                continue
            array = np.stack(values)
            norms = np.linalg.norm(array, axis=1)
            stats[key] = {
                "count": len(values),
                "dimension": array.shape[1] if array.ndim > 1 else 0,
                "mean_norm": float(np.mean(norms)),
                "std_norm": float(np.std(norms)),
            }
        return stats


def create_embedding_aware_system(
    concept_graph: "ConceptGraph",
    cosine_alpha: float = 0.5,
    google_base_frequency: int = 1000,
    hybrid_weight: float = 0.5,
    cache_results: bool = True,
    ensure_embeddings: bool = True,
    search_provider: WebSearchProvider | None = None,
    provider_seed: int | None = 0,
) -> EmbeddingAwareSimilaritySystem:
    """Create an embedding-aware system with an offline provider by default."""
    return EmbeddingAwareSimilaritySystem(
        concept_graph=concept_graph,
        cosine_alpha=cosine_alpha,
        google_base_frequency=google_base_frequency,
        hybrid_weight=hybrid_weight,
        cache_results=cache_results,
        ensure_embeddings=ensure_embeddings,
        search_provider=search_provider,
        provider_seed=provider_seed,
    )


def analyze_alignment_similarity(
    graph: "ConceptGraph",
    system: SimilarityComparisonSystem,
    positive_gold: set[tuple[str, str]],
    *,
    similarity_mode: SimilarityMethod = SimilarityMethod.HYBRID,
    similarity_threshold: float | None = None,
    gold_by_source: dict[str, set[str]] | None = None,
    top_ids_by_source: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Summarize gold-pair scores and optional per-source threshold predictions."""
    if gold_by_source is None:
        gold_by_source = {}
        for source_id, target_id in positive_gold:
            gold_by_source.setdefault(source_id, set()).add(target_id)

    score_index = {
        SimilarityMethod.COSINE: 0,
        SimilarityMethod.GOOGLE_INDEX: 1,
        SimilarityMethod.HYBRID: 2,
    }[similarity_mode]
    predicted: set[tuple[str, str]] = set()
    if similarity_threshold is not None and top_ids_by_source:
        for source_id, candidate_ids in top_ids_by_source.items():
            if source_id not in graph.nodes:
                continue
            for candidate_id in candidate_ids:
                if candidate_id not in graph.nodes:
                    continue
                scores = system.compute_similarity(
                    graph,
                    source_id,
                    candidate_id,
                    SimilarityMethod.HYBRID,
                    return_all=True,
                )
                if scores[score_index] >= similarity_threshold:
                    predicted.add((source_id, candidate_id))

    totals = [0.0, 0.0, 0.0]
    count = 0
    for source_id, target_ids in gold_by_source.items():
        if source_id not in graph.nodes:
            continue
        for target_id in target_ids:
            if target_id not in graph.nodes:
                continue
            scores = system.compute_similarity(
                graph,
                source_id,
                target_id,
                SimilarityMethod.HYBRID,
                return_all=True,
            )
            totals = [total + float(score) for total, score in zip(totals, scores)]
            count += 1

    result: dict[str, Any] = {
        "threshold_mode": similarity_mode,
        "similarity_avgs_over_gold": {
            name: total / count if count else 0.0
            for name, total in zip(("cosine", "google", "hybrid"), totals)
        },
    }
    if similarity_threshold is not None:
        true_positives = len(positive_gold & predicted)
        precision = true_positives / len(predicted) if predicted else 0.0
        recall = true_positives / len(positive_gold) if positive_gold else 0.0
        result["threshold_metrics"] = {
            "threshold": similarity_threshold,
            "precision": precision,
            "recall": recall,
            "f1": (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            ),
            "TP": true_positives,
            "|predicted|": len(predicted),
        }
    return result


def analyze_similarity_methods_with_views(
    concept_graph: "ConceptGraph",
    hypergraph_view: "HypergraphView | None" = None,
    multigraph_view: "MultigraphView | None" = None,
    k: int = 10,
    **kwargs: Any,
) -> dict[str, Any]:
    """Compare top pairs and summarize optional graph views."""
    system = create_embedding_aware_system(concept_graph, **kwargs)
    comparisons = system.compare_top_k_pairs(concept_graph, k=k)
    integration_results = system.integrate_with_views(
        concept_graph, hypergraph_view, multigraph_view
    )
    return {
        "top_k_comparisons": [
            {
                "node_id_1": c.node_id_1,
                "node_id_2": c.node_id_2,
                "cosine_similarity": c.cosine_similarity,
                "google_similarity": c.google_similarity,
                "hybrid_similarity": c.hybrid_similarity,
                "cosine_rank": c.cosine_rank,
                "google_rank": c.google_rank,
                "hybrid_rank": c.hybrid_rank
            }
            for c in comparisons
        ],
        "integration_results": integration_results,
        "system_config": {
            "cosine_alpha": system.cosine_plugin.alpha,
            "hybrid_weight": system.hybrid_weight,
            "cache_results": system.cache_results,
        },
    }
