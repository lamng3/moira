"""
Similarity comparison system that integrates Google index and cosine similarity plugins.
Provides unified interface for comparing similarity methods and finding top-k pairs.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from agentoi.algorithms.graph import ConceptGraph
    from agentoi.algorithms.views import HypergraphView, MultigraphView

from .cosine import CosineSimilarityPlugin
from .engine import NodePairSimilarityEngine
from .ngd import NormalizedGoogleDistance
from agentoi.retrieval.web import StaticSearchProvider
from .types import (
    ComparisonResult,
    SimilarityMethod,
    SimilarityResult,
)

class SimilarityComparisonSystem:
    """
    Unified similarity comparison system that integrates multiple similarity methods.
    
    Supports:
    - Cosine similarity (embedding-based)
    - Google index similarity (NGD-based)
    - Hybrid approaches
    - Integration with ConceptGraph, HypergraphView, and MultigraphView
    - Top-k pair retrieval and comparison
    """
    
    def __init__(
        self,
        cosine_plugin: Optional[CosineSimilarityPlugin] = None,
        ngd: Optional[NormalizedGoogleDistance] = None,
        hybrid_weight: float = 0.5,
        cache_results: bool = True
    ):
        """
        Initialize the similarity comparison system.
        
        Args:
            cosine_plugin: Cosine similarity plugin (defaults to new instance)
            ngd: Normalized Google Distance scorer.
            hybrid_weight: Weight for hybrid similarity (0.0 = cosine only, 1.0 = google only)
            cache_results: Whether to cache similarity results
        """
        self.engine = NodePairSimilarityEngine(
            cosine_plugin=cosine_plugin,
            ngd=ngd,
            hybrid_weight=hybrid_weight,
        )
        self.cosine_plugin = self.engine.cosine_plugin
        self.ngd = self.engine.ngd
        self.hybrid_weight = self.engine.hybrid_weight
        self.cache_results = cache_results
        
        # Caching
        self._similarity_cache: Dict[
            tuple[Any, ...],
            Union[float, Tuple[float, float, float]],
        ] = {}
    
    def _get_cache_key(
        self,
        graph: "ConceptGraph",
        node_id_1: str,
        node_id_2: str,
        method: str,
    ) -> tuple[Any, ...]:
        """Generate cache key for similarity computation."""
        first, second = sorted(
            (
                self.engine.node_signature(graph.nodes[node_id_1]),
                self.engine.node_signature(graph.nodes[node_id_2]),
            )
        )
        return (id(graph), first, second, method)
    
    def compute_similarity(
        self,
        graph: "ConceptGraph",
        node_id_1: str,
        node_id_2: str,
        method: SimilarityMethod,
        return_all: bool = False,
    ) -> Union[float, Tuple[float, float, float]]:
        """
        Compute similarity between two nodes using specified method.
        
        Args:
            graph: The concept graph
            node_id_1: First node ID
            node_id_2: Second node ID
            method: Similarity method to use
        
        Returns:
            float: Similarity score between 0 and 1
        """
        if self.cache_results:
            cache_method = (
                f"{method.value}:all"
                if method == SimilarityMethod.HYBRID and return_all
                else method.value
            )
            cache_key = self._get_cache_key(
                graph, node_id_1, node_id_2, cache_method
            )
            if cache_key in self._similarity_cache:
                return self._similarity_cache[cache_key]
        
        similarity = self.engine.score(
            graph.nodes[node_id_1],
            graph.nodes[node_id_2],
            method,
            return_all=return_all,
        )
        
        # Cache result
        if self.cache_results:
            self._similarity_cache[cache_key] = similarity
        
        return similarity
    
    def compare_similarities(
        self,
        graph: "ConceptGraph",
        node_id_1: str,
        node_id_2: str
    ) -> ComparisonResult:
        """
        Compare all similarity methods for a pair of nodes.
        
        Args:
            graph: The concept graph
            node_id_1: First node ID
            node_id_2: Second node ID
        
        Returns:
            ComparisonResult: Comparison of all similarity methods
        """
        cosine_sim, google_sim, hybrid_sim = self.compute_similarity(
            graph,
            node_id_1,
            node_id_2,
            SimilarityMethod.HYBRID,
            return_all=True,
        )
        
        # Create result (ranks will be computed later)
        result = ComparisonResult(
            node_id_1=node_id_1,
            node_id_2=node_id_2,
            cosine_similarity=cosine_sim,
            google_similarity=google_sim,
            hybrid_similarity=hybrid_sim,
            cosine_rank=0,  # Will be set later
            google_rank=0,  # Will be set later
            hybrid_rank=0,  # Will be set later
            metadata={
                "hybrid_weight": self.hybrid_weight,
                "cosine_alpha": self.cosine_plugin.alpha,
                "ngd_min_count": self.ngd.min_count
            }
        )
        
        return result
    
    def get_top_k_pairs(
        self,
        graph: "ConceptGraph",
        k: int = 10,
        method: SimilarityMethod = SimilarityMethod.COSINE,
        node_ids: Optional[List[str]] = None,
        exclude_self: bool = True
    ) -> List[SimilarityResult]:
        """
        Get top-k most similar node pairs using specified method.
        
        Args:
            graph: The concept graph
            k: Number of top pairs to return
            method: Similarity method to use
            node_ids: Optional list of specific node IDs to consider
            exclude_self: Whether to exclude self-similarity
        
        Returns:
            List of SimilarityResult objects
        """
        if node_ids is None:
            node_ids = list(graph.nodes.keys())
        node_ids = [node_id for node_id in node_ids if node_id in graph.nodes]
        
        # Compute all pairwise similarities
        similarities = []
        for i, node_id_1 in enumerate(node_ids):
            start = i + 1 if exclude_self else i
            for node_id_2 in node_ids[start:]:
                similarity = self.compute_similarity(
                    graph, node_id_1, node_id_2, method
                )
                similarities.append(SimilarityResult(
                    method=method,
                    node_id_1=node_id_1,
                    node_id_2=node_id_2,
                    similarity_score=similarity
                ))
        
        # Sort by similarity score and return top-k
        similarities.sort(key=lambda x: x.similarity_score, reverse=True)
        return similarities[:k]
    
    def compare_top_k_pairs(
        self,
        graph: "ConceptGraph",
        k: int = 10,
        node_ids: Optional[List[str]] = None,
        exclude_self: bool = True
    ) -> List[ComparisonResult]:
        """
        Compare top-k pairs across all similarity methods.
        
        Args:
            graph: The concept graph
            k: Number of top pairs to return
            node_ids: Optional list of specific node IDs to consider
            exclude_self: Whether to exclude self-similarity
        
        Returns:
            List of ComparisonResult objects with rankings
        """
        if node_ids is None:
            node_ids = list(graph.nodes.keys())
        node_ids = [node_id for node_id in node_ids if node_id in graph.nodes]
        
        # Get all possible pairs
        pairs = []
        for i, node_id_1 in enumerate(node_ids):
            start = i + 1 if exclude_self else i
            pairs.extend((node_id_1, node_id_2) for node_id_2 in node_ids[start:])
        
        # Compute comparisons for all pairs
        comparisons = []
        for node_id_1, node_id_2 in pairs:
            comparison = self.compare_similarities(graph, node_id_1, node_id_2)
            comparisons.append(comparison)
        
        # Sort by each method and assign ranks
        rank_fields = {
            SimilarityMethod.COSINE: ("cosine_similarity", "cosine_rank"),
            SimilarityMethod.GOOGLE_INDEX: ("google_similarity", "google_rank"),
            SimilarityMethod.HYBRID: ("hybrid_similarity", "hybrid_rank"),
        }
        for score_field, rank_field in rank_fields.values():
            comparisons.sort(
                key=lambda comparison: getattr(comparison, score_field),
                reverse=True,
            )
            for rank, comparison in enumerate(comparisons, start=1):
                setattr(comparison, rank_field, rank)
        
        # Sort by hybrid similarity for final ranking
        comparisons.sort(key=lambda x: x.hybrid_similarity, reverse=True)
        return comparisons[:k]
    
    def analyze_similarity_methods(
        self,
        graph: "ConceptGraph",
        node_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze the correlation and differences between similarity methods.
        
        Args:
            graph: The concept graph
            node_ids: Optional list of specific node IDs to consider
        
        Returns:
            Dictionary with analysis results
        """
        if node_ids is None:
            node_ids = list(graph.nodes.keys())
        
        # Get all comparisons
        comparisons = self.compare_top_k_pairs(graph, k=len(node_ids) * (len(node_ids) - 1) // 2, node_ids=node_ids)
        if not comparisons:
            empty_stats = {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
            return {
                "num_pairs": 0,
                "correlations": {
                    "cosine_google": 0.0,
                    "cosine_hybrid": 0.0,
                    "google_hybrid": 0.0,
                },
                "statistics": {
                    "cosine": dict(empty_stats),
                    "google": dict(empty_stats),
                    "hybrid": dict(empty_stats),
                },
            }
        
        # Extract similarity scores
        cosine_scores = [c.cosine_similarity for c in comparisons]
        google_scores = [c.google_similarity for c in comparisons]
        hybrid_scores = [c.hybrid_similarity for c in comparisons]
        
        # Compute correlations
        cosine_google_corr = np.corrcoef(cosine_scores, google_scores)[0, 1] if len(cosine_scores) > 1 else 0.0
        cosine_hybrid_corr = np.corrcoef(cosine_scores, hybrid_scores)[0, 1] if len(cosine_scores) > 1 else 0.0
        google_hybrid_corr = np.corrcoef(google_scores, hybrid_scores)[0, 1] if len(google_scores) > 1 else 0.0
        
        # Compute statistics
        analysis = {
            "num_pairs": len(comparisons),
            "correlations": {
                "cosine_google": cosine_google_corr,
                "cosine_hybrid": cosine_hybrid_corr,
                "google_hybrid": google_hybrid_corr
            },
            "statistics": {
                "cosine": {
                    "mean": np.mean(cosine_scores),
                    "std": np.std(cosine_scores),
                    "min": np.min(cosine_scores),
                    "max": np.max(cosine_scores)
                },
                "google": {
                    "mean": np.mean(google_scores),
                    "std": np.std(google_scores),
                    "min": np.min(google_scores),
                    "max": np.max(google_scores)
                },
                "hybrid": {
                    "mean": np.mean(hybrid_scores),
                    "std": np.std(hybrid_scores),
                    "min": np.min(hybrid_scores),
                    "max": np.max(hybrid_scores)
                }
            }
        }
        
        return analysis
    
    def integrate_with_views(
        self,
        graph: "ConceptGraph",
        hypergraph_view: Optional["HypergraphView"] = None,
        multigraph_view: Optional["MultigraphView"] = None
    ) -> Dict[str, Any]:
        """
        Integrate similarity comparison with HypergraphView and MultigraphView.
        
        Args:
            graph: The concept graph
            hypergraph_view: Optional hypergraph view
            multigraph_view: Optional multigraph view
        
        Returns:
            Dictionary with integration results
        """
        results = {}
        
        # Get node IDs from views if available
        node_ids = None
        if hypergraph_view:
            node_ids = list(hypergraph_view.nodes)
            results["hypergraph_nodes"] = len(hypergraph_view.nodes)
            results["hypergraph_hyperedges"] = len(hypergraph_view.hyperedges)
        
        if multigraph_view:
            if node_ids is None:
                node_ids = list(multigraph_view.node_attrs.keys())
            results["multigraph_nodes"] = len(multigraph_view.node_attrs)
            results["multigraph_edges"] = len(multigraph_view.edges)
        
        # Compute similarity analysis
        if node_ids:
            analysis = self.analyze_similarity_methods(graph, node_ids)
            results.update(analysis)
        
        return results


# Convenience functions
def create_similarity_comparison_system(
    cosine_alpha: float = 0.5,
    google_mock_data: Optional[Dict[str, int]] = None,
    hybrid_weight: float = 0.5,
    cache_results: bool = True
) -> SimilarityComparisonSystem:
    """
    Create a similarity comparison system with default settings.
    
    Args:
        cosine_alpha: Alpha parameter for cosine similarity
        google_mock_data: Mock data for Google search provider
        hybrid_weight: Weight for hybrid similarity
        cache_results: Whether to cache results
    
    Returns:
        SimilarityComparisonSystem: Configured comparison system
    """
    cosine_plugin = CosineSimilarityPlugin(alpha=cosine_alpha)
    ngd = NormalizedGoogleDistance(
        search_provider=StaticSearchProvider(counts=google_mock_data)
    )
    
    return SimilarityComparisonSystem(
        cosine_plugin=cosine_plugin,
        ngd=ngd,
        hybrid_weight=hybrid_weight,
        cache_results=cache_results
    )


def compare_similarity_methods(
    graph: "ConceptGraph",
    k: int = 10,
    node_ids: Optional[List[str]] = None,
    **kwargs
) -> List[ComparisonResult]:
    """
    Convenience function to compare similarity methods.
    
    Args:
        graph: The concept graph
        k: Number of top pairs to return
        node_ids: Optional list of specific node IDs to consider
        **kwargs: Additional arguments for SimilarityComparisonSystem
    
    Returns:
        List of ComparisonResult objects
    """
    system = create_similarity_comparison_system(**kwargs)
    return system.compare_top_k_pairs(graph, k=k, node_ids=node_ids)
