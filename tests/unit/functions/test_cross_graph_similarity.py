"""
Test cases for cross-graph similarity computation.
Tests finding top-k most similar node pairs between two different concept graphs.
"""

import torch
import pytest
from typing import List, Dict
from unittest.mock import Mock, patch

from moira.functions.similarity import (
    CrossGraphSimilarity,
    CrossGraphPair,
    SimilarityMethod,
    find_top_k_cross_graph_pairs,
    compare_cross_graph_methods,
    analyze_cross_graph_similarity
)
from moira.algorithms.graph import ConceptGraph, EquivalentClass, Concept


class TestCrossGraphSimilarity:
    """Test cases for CrossGraphSimilarity."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create concepts for graph A
        self.concepts_a = [
            Concept(
                nodeid="concept_a_1",
                name="Animal",
                ground_set={
                    "labels": ["Animal", "Creature"],
                    "exact_synonyms": ["Beast", "Fauna"]
                }
            ),
            Concept(
                nodeid="concept_a_2",
                name="Plant",
                ground_set={
                    "labels": ["Plant", "Flora"],
                    "exact_synonyms": ["Vegetation", "Botanical"]
                }
            ),
            Concept(
                nodeid="concept_a_3",
                name="Water",
                ground_set={
                    "labels": ["Water", "Liquid"],
                    "exact_synonyms": ["H2O", "Aquatic"]
                }
            )
        ]
        
        # Create concepts for graph B
        self.concepts_b = [
            Concept(
                nodeid="concept_b_1",
                name="Living organism",
                ground_set={
                    "labels": ["Living organism", "Life form"],
                    "exact_synonyms": ["Biotic entity", "Living being"]
                }
            ),
            Concept(
                nodeid="concept_b_2",
                name="Vegetation",
                ground_set={
                    "labels": ["Vegetation", "Plant life"],
                    "exact_synonyms": ["Flora", "Botanical life"]
                }
            ),
            Concept(
                nodeid="concept_b_3",
                name="Liquid",
                ground_set={
                    "labels": ["Liquid", "Fluid"],
                    "exact_synonyms": ["Water", "Aqueous"]
                }
            ),
            Concept(
                nodeid="concept_b_4",
                name="Mineral",
                ground_set={
                    "labels": ["Mineral", "Rock"],
                    "exact_synonyms": ["Inorganic", "Geological"]
                }
            )
        ]
        
        # Create equivalent classes for graph A
        self.equiv_classes_a = []
        for i, concept in enumerate(self.concepts_a):
            equiv_class = EquivalentClass([concept])
            equiv_class.id = f"equiv_a_{i+1}"
            self.equiv_classes_a.append(equiv_class)
        
        # Create equivalent classes for graph B
        self.equiv_classes_b = []
        for i, concept in enumerate(self.concepts_b):
            equiv_class = EquivalentClass([concept])
            equiv_class.id = f"equiv_b_{i+1}"
            self.equiv_classes_b.append(equiv_class)
        
        # Create concept graphs
        self.graph_a = ConceptGraph(nodes=self.equiv_classes_a, edges=[])
        self.graph_b = ConceptGraph(nodes=self.equiv_classes_b, edges=[])
        
        # Mock embeddings
        for node in self.graph_a.nodes.values():
            node.embedding = torch.randn(768)
            node.text_embedding = torch.randn(768)
            node.graph_embedding = torch.randn(768)
        
        for node in self.graph_b.nodes.values():
            node.embedding = torch.randn(768)
            node.text_embedding = torch.randn(768)
            node.graph_embedding = torch.randn(768)
        
        # Initialize similarity system
        self.similarity_system = CrossGraphSimilarity()
    
    def test_initialization(self):
        """Test that CrossGraphSimilarity initializes correctly."""
        system = CrossGraphSimilarity(
            hybrid_weight=0.7,
            cache_results=True
        )
        assert system.hybrid_weight == 0.7
        assert system.cache_results == True
        assert isinstance(system.cosine_plugin, type(self.similarity_system.cosine_plugin))
        assert isinstance(system.ngd, type(self.similarity_system.ngd))
    
    def test_cosine_similarity_computation(self):
        """Test cosine similarity computation between nodes from different graphs."""
        node_a = self.graph_a.nodes["equiv_a_1"]
        node_b = self.graph_b.nodes["equiv_b_1"]
        
        similarity = self.similarity_system.compute_cross_graph_similarity(
            node_a, node_b, SimilarityMethod.COSINE
        )
        
        assert 0.0 <= similarity <= 1.0
        assert isinstance(similarity, float)
    
    def test_google_similarity_computation(self):
        """Test Google index similarity computation between nodes from different graphs."""
        node_a = self.graph_a.nodes["equiv_a_1"]
        node_b = self.graph_b.nodes["equiv_b_1"]
        
        similarity = self.similarity_system.compute_cross_graph_similarity(
            node_a, node_b, SimilarityMethod.GOOGLE_INDEX
        )
        
        assert 0.0 <= similarity <= 1.0
        assert isinstance(similarity, float)
    
    def test_hybrid_similarity_computation(self):
        """Test hybrid similarity computation between nodes from different graphs."""
        node_a = self.graph_a.nodes["equiv_a_1"]
        node_b = self.graph_b.nodes["equiv_b_1"]
        
        similarity = self.similarity_system.compute_cross_graph_similarity(
            node_a, node_b, SimilarityMethod.HYBRID
        )
        
        assert 0.0 <= similarity <= 1.0
        assert isinstance(similarity, float)
    
    def test_cross_graph_similarity_computation(self):
        """Test cross-graph similarity computation."""
        node_a = self.graph_a.nodes["equiv_a_1"]
        node_b = self.graph_b.nodes["equiv_b_1"]
        
        # Test cosine similarity
        cosine_sim = self.similarity_system.compute_cross_graph_similarity(
            node_a, node_b, SimilarityMethod.COSINE
        )
        assert 0.0 <= cosine_sim <= 1.0
        
        # Test Google similarity
        google_sim = self.similarity_system.compute_cross_graph_similarity(
            node_a, node_b, SimilarityMethod.GOOGLE_INDEX
        )
        assert 0.0 <= google_sim <= 1.0
        
        # Test hybrid similarity
        hybrid_sim = self.similarity_system.compute_cross_graph_similarity(
            node_a, node_b, SimilarityMethod.HYBRID
        )
        assert 0.0 <= hybrid_sim <= 1.0
    
    def test_find_top_k_cross_graph_pairs(self):
        """Test finding top-k cross-graph pairs."""
        # Test with cosine similarity
        top_pairs = self.similarity_system.find_top_k_cross_graph_pairs(
            graph_a=self.graph_a,
            graph_b=self.graph_b,
            k=3,
            method=SimilarityMethod.COSINE,
            similarity_threshold=0.0
        )
        
        assert len(top_pairs) <= 3
        assert all(isinstance(pair, CrossGraphPair) for pair in top_pairs)
        
        # Check that pairs are sorted by similarity score
        if len(top_pairs) > 1:
            for i in range(len(top_pairs) - 1):
                assert top_pairs[i].similarity_score >= top_pairs[i + 1].similarity_score
        
        # Check pair properties
        for pair in top_pairs:
            assert pair.node_a_id in self.graph_a.nodes
            assert pair.node_b_id in self.graph_b.nodes
            assert 0.0 <= pair.similarity_score <= 1.0
            assert pair.method == "cosine"
    
    def test_find_top_k_cross_graph_pairs_with_threshold(self):
        """Test finding top-k pairs with similarity threshold."""
        # Test with high threshold
        top_pairs = self.similarity_system.find_top_k_cross_graph_pairs(
            graph_a=self.graph_a,
            graph_b=self.graph_b,
            k=10,
            method=SimilarityMethod.COSINE,
            similarity_threshold=0.9  # Very high threshold
        )
        
        # Should have fewer pairs due to high threshold
        assert len(top_pairs) <= 10
        assert all(pair.similarity_score >= 0.9 for pair in top_pairs)
    
    def test_find_top_k_cross_graph_pairs_with_node_subset(self):
        """Test finding top-k pairs with specific node subsets."""
        # Test with specific node IDs
        node_ids_a = ["equiv_a_1", "equiv_a_2"]
        node_ids_b = ["equiv_b_1", "equiv_b_2"]
        
        top_pairs = self.similarity_system.find_top_k_cross_graph_pairs(
            graph_a=self.graph_a,
            graph_b=self.graph_b,
            k=5,
            method=SimilarityMethod.COSINE,
            node_ids_a=node_ids_a,
            node_ids_b=node_ids_b
        )
        
        # Should have at most 4 pairs (2 x 2)
        assert len(top_pairs) <= 4
        assert all(pair.node_a_id in node_ids_a for pair in top_pairs)
        assert all(pair.node_b_id in node_ids_b for pair in top_pairs)
    
    def test_compare_cross_graph_methods(self):
        """Test comparing all similarity methods."""
        comparisons = self.similarity_system.compare_cross_graph_methods(
            graph_a=self.graph_a,
            graph_b=self.graph_b,
            k=2,
            similarity_threshold=0.0
        )
        
        assert "cosine" in comparisons
        assert "google_index" in comparisons
        assert "hybrid" in comparisons
        
        for method_name, pairs in comparisons.items():
            assert len(pairs) <= 2
            assert all(isinstance(pair, CrossGraphPair) for pair in pairs)
            assert all(pair.method == method_name for pair in pairs)
    
    def test_analyze_cross_graph_similarity(self):
        """Test cross-graph similarity analysis."""
        analysis = self.similarity_system.analyze_cross_graph_similarity(
            graph_a=self.graph_a,
            graph_b=self.graph_b,
            method=SimilarityMethod.HYBRID
        )
        
        assert "total_pairs" in analysis
        assert "method_used" in analysis
        assert "statistics" in analysis
        assert "score_distribution" in analysis
        assert "top_pairs" in analysis
        
        assert analysis["total_pairs"] > 0
        assert analysis["method_used"] == "hybrid"
        
        # Check statistics
        stats = analysis["statistics"]
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "median" in stats
        
        # Check score distribution
        dist = analysis["score_distribution"]
        assert "high_similarity" in dist
        assert "medium_similarity" in dist
        assert "low_similarity" in dist
        
        # Check top pairs
        assert len(analysis["top_pairs"]) > 0
        for pair_info in analysis["top_pairs"]:
            assert "node_a_id" in pair_info
            assert "node_b_id" in pair_info
            assert "similarity_score" in pair_info
    
    def test_caching_behavior(self):
        """Test that caching works correctly."""
        node_a = self.graph_a.nodes["equiv_a_1"]
        node_b = self.graph_b.nodes["equiv_b_1"]
        
        # First computation
        sim1 = self.similarity_system.compute_cross_graph_similarity(
            node_a, node_b, SimilarityMethod.COSINE
        )
        
        # Second computation should use cache
        sim2 = self.similarity_system.compute_cross_graph_similarity(
            node_a, node_b, SimilarityMethod.COSINE
        )
        
        assert sim1 == sim2
        
        # Check that cache was populated
        cache_key = self.similarity_system._get_cache_key(
            node_a, node_b, "cosine"
        )
        assert cache_key in self.similarity_system._similarity_cache

    def test_cache_distinguishes_different_nodes_with_the_same_ids(self):
        """Node IDs are only graph-local and must not alias cache entries."""
        node_a = Mock(id="shared")
        node_b_1 = Mock(id="shared")
        node_b_2 = Mock(id="shared")
        system = CrossGraphSimilarity()
        system.engine.score = Mock(side_effect=[0.25, 0.75])

        first = system.compute_cross_graph_similarity(
            node_a, node_b_1, SimilarityMethod.COSINE
        )
        second = system.compute_cross_graph_similarity(
            node_a, node_b_2, SimilarityMethod.COSINE
        )

        assert first == 0.25
        assert second == 0.75
        assert system.engine.score.call_count == 2
        assert len(system._similarity_cache) == 2

    def test_cache_key_preserves_cross_graph_direction(self):
        node_a = Mock(id="same")
        node_b = Mock(id="same")
        system = CrossGraphSimilarity()

        assert system._get_cache_key(
            node_a, node_b, "cosine"
        ) != system._get_cache_key(node_b, node_a, "cosine")
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with empty graphs
        empty_graph = ConceptGraph(nodes=[], edges=[])
        
        top_pairs = self.similarity_system.find_top_k_cross_graph_pairs(
            graph_a=empty_graph,
            graph_b=self.graph_b,
            k=5,
            method=SimilarityMethod.COSINE
        )
        
        assert len(top_pairs) == 0
        
        # Test with invalid node IDs
        top_pairs = self.similarity_system.find_top_k_cross_graph_pairs(
            graph_a=self.graph_a,
            graph_b=self.graph_b,
            k=5,
            method=SimilarityMethod.COSINE,
            node_ids_a=["nonexistent"],
            node_ids_b=["nonexistent"]
        )
        
        assert len(top_pairs) == 0


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create simple test graphs
        self.concept_a = Concept(
            nodeid="concept_a",
            name="Test A",
            ground_set={"labels": ["Test A"]}
        )
        self.concept_b = Concept(
            nodeid="concept_b",
            name="Test B",
            ground_set={"labels": ["Test B"]}
        )
        
        self.equiv_a = EquivalentClass([self.concept_a])
        self.equiv_a.id = "equiv_a"
        self.equiv_b = EquivalentClass([self.concept_b])
        self.equiv_b.id = "equiv_b"
        
        self.graph_a = ConceptGraph(nodes=[self.equiv_a], edges=[])
        self.graph_b = ConceptGraph(nodes=[self.equiv_b], edges=[])
        
        # Mock embeddings
        self.equiv_a.embedding = torch.randn(768)
        self.equiv_b.embedding = torch.randn(768)
    
    def test_find_top_k_cross_graph_pairs_convenience(self):
        """Test convenience function for finding top-k pairs."""
        top_pairs = find_top_k_cross_graph_pairs(
            graph_a=self.graph_a,
            graph_b=self.graph_b,
            k=1,
            method=SimilarityMethod.COSINE
        )
        
        assert len(top_pairs) <= 1
        assert all(isinstance(pair, CrossGraphPair) for pair in top_pairs)
    
    def test_compare_cross_graph_methods_convenience(self):
        """Test convenience function for comparing methods."""
        comparisons = compare_cross_graph_methods(
            graph_a=self.graph_a,
            graph_b=self.graph_b,
            k=1
        )
        
        assert "cosine" in comparisons
        assert "google_index" in comparisons
        assert "hybrid" in comparisons
    
    def test_analyze_cross_graph_similarity_convenience(self):
        """Test convenience function for analysis."""
        analysis = analyze_cross_graph_similarity(
            graph_a=self.graph_a,
            graph_b=self.graph_b,
            method=SimilarityMethod.HYBRID
        )
        
        assert "total_pairs" in analysis
        assert "method_used" in analysis


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
