"""
Test cases for the cosine similarity plugin integration with concept graph.
"""

import torch
import pytest
from typing import List, Dict
from unittest.mock import Mock, patch

from agentoi.functions.similarity import (
    CosineSimilarityPlugin, 
    compute_cosine_similarity_matrix,
    compute_similarity_to_query
)
from agentoi.algorithms.graph import ConceptGraph, EquivalentClass, Concept


class TestCosineSimilarityPlugin:
    """Test cases for CosineSimilarityPlugin integration with ConceptGraph."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create mock concepts
        self.concept1 = Concept(
            nodeid="concept1",
            name="Animal",
            ground_set={
                "labels": ["Animal", "Creature"],
                "exact_synonyms": ["Beast", "Fauna"]
            }
        )
        
        self.concept2 = Concept(
            nodeid="concept2", 
            name="Dog",
            ground_set={
                "labels": ["Dog", "Canine"],
                "exact_synonyms": ["Puppy", "Hound"]
            }
        )
        
        self.concept3 = Concept(
            nodeid="concept3",
            name="Cat", 
            ground_set={
                "labels": ["Cat", "Feline"],
                "exact_synonyms": ["Kitten", "Moggy"]
            }
        )
        
        # Create equivalent classes
        self.equiv_class1 = EquivalentClass([self.concept1])
        self.equiv_class1.id = "equiv1"
        
        self.equiv_class2 = EquivalentClass([self.concept2])
        self.equiv_class2.id = "equiv2"
        
        self.equiv_class3 = EquivalentClass([self.concept3])
        self.equiv_class3.id = "equiv3"
        
        # Mock embeddings for testing
        self.mock_embedding1 = torch.tensor([1.0, 0.0, 0.0])
        self.mock_embedding2 = torch.tensor([0.8, 0.6, 0.0])  # Similar to embedding1
        self.mock_embedding3 = torch.tensor([0.0, 0.0, 1.0])  # Different from embedding1
        
        # Create concept graph
        self.concept_graph = ConceptGraph(
            nodes=[self.equiv_class1, self.equiv_class2, self.equiv_class3],
            edges=[]
        )
        
        # Initialize plugin
        self.plugin = CosineSimilarityPlugin(alpha=0.5, normalize=True)
    
    def test_plugin_initialization(self):
        """Test that the plugin initializes correctly."""
        plugin = CosineSimilarityPlugin(alpha=0.7, normalize=False)
        assert plugin.alpha == 0.7
        assert plugin.normalize == False
    
    @patch('agentoi.algorithms.graph.EquivalentClass.compute_embedding')
    def test_get_node_embeddings(self, mock_compute_embedding):
        """Test extraction of node embeddings from concept graph."""
        # Mock the compute_embedding method
        def mock_embedding(node, alpha=0.5):
            if node.id == "equiv1":
                return self.mock_embedding1
            elif node.id == "equiv2":
                return self.mock_embedding2
            elif node.id == "equiv3":
                return self.mock_embedding3
        
        mock_compute_embedding.side_effect = mock_embedding
        
        # Set up mock embeddings on nodes
        self.equiv_class1.embedding = self.mock_embedding1
        self.equiv_class2.embedding = self.mock_embedding2
        self.equiv_class3.embedding = self.mock_embedding3
        
        # Test getting all embeddings
        embeddings, node_ids = self.plugin._get_node_embeddings(self.concept_graph)
        
        assert embeddings.shape == (3, 3)
        assert len(node_ids) == 3
        assert "equiv1" in node_ids
        assert "equiv2" in node_ids
        assert "equiv3" in node_ids
    
    @patch('agentoi.algorithms.graph.EquivalentClass.compute_embedding')
    def test_get_node_embeddings_subset(self, mock_compute_embedding):
        """Test extraction of embeddings for a subset of nodes."""
        # Set up mock embeddings
        self.equiv_class1.embedding = self.mock_embedding1
        self.equiv_class2.embedding = self.mock_embedding2
        self.equiv_class3.embedding = self.mock_embedding3
        
        # Test getting subset of embeddings
        subset_ids = ["equiv1", "equiv3"]
        embeddings, node_ids = self.plugin._get_node_embeddings(
            self.concept_graph, subset_ids
        )
        
        assert embeddings.shape == (2, 3)
        assert node_ids == ["equiv1", "equiv3"]
    
    @patch('agentoi.algorithms.graph.EquivalentClass.compute_embedding')
    def test_compute_pairwise_similarity(self, mock_compute_embedding):
        """Test pairwise similarity matrix computation."""
        # Set up mock embeddings
        self.equiv_class1.embedding = self.mock_embedding1
        self.equiv_class2.embedding = self.mock_embedding2
        self.equiv_class3.embedding = self.mock_embedding3
        
        # Compute similarity matrix
        similarity_matrix = self.plugin.compute_pairwise_similarity(self.concept_graph)
        
        # Check matrix properties
        assert similarity_matrix.shape == (3, 3)
        assert torch.allclose(similarity_matrix, similarity_matrix.T)  # Symmetric
        
        # Check diagonal elements (self-similarity should be 1.0 after normalization)
        diagonal = torch.diag(similarity_matrix)
        assert torch.allclose(diagonal, torch.ones(3), atol=1e-6)
    
    @patch('agentoi.algorithms.graph.EquivalentClass.compute_embedding')
    def test_compute_similarity_to_query(self, mock_compute_embedding):
        """Test similarity computation between query and nodes."""
        # Set up mock embeddings
        self.equiv_class1.embedding = self.mock_embedding1
        self.equiv_class2.embedding = self.mock_embedding2
        self.equiv_class3.embedding = self.mock_embedding3
        
        # Create query embedding similar to embedding1
        query_embedding = torch.tensor([0.9, 0.1, 0.0])
        
        # Compute similarities
        similarities = self.plugin.compute_similarity_to_query(
            self.concept_graph, query_embedding
        )
        
        assert similarities.shape == (3,)
        # Query should be most similar to equiv1 (which has embedding1)
        assert similarities[0] > similarities[1]  # equiv1 > equiv2
        assert similarities[0] > similarities[2]  # equiv1 > equiv3
    
    @patch('agentoi.algorithms.graph.EquivalentClass.compute_embedding')
    def test_get_top_similar_pairs(self, mock_compute_embedding):
        """Test getting top similar node pairs."""
        # Set up mock embeddings
        self.equiv_class1.embedding = self.mock_embedding1
        self.equiv_class2.embedding = self.mock_embedding2
        self.equiv_class3.embedding = self.mock_embedding3
        
        # Get top similar pairs
        top_pairs = self.plugin.get_top_similar_pairs(self.concept_graph, k=2)
        
        assert len(top_pairs) <= 2
        for node_id_1, node_id_2, similarity in top_pairs:
            assert node_id_1 in ["equiv1", "equiv2", "equiv3"]
            assert node_id_2 in ["equiv1", "equiv2", "equiv3"]
            assert node_id_1 != node_id_2  # No self-pairs
            assert 0.0 <= similarity <= 1.0
    
    @patch('agentoi.algorithms.graph.EquivalentClass.compute_embedding')
    def test_get_most_similar_nodes(self, mock_compute_embedding):
        """Test getting most similar nodes to a query."""
        # Set up mock embeddings
        self.equiv_class1.embedding = self.mock_embedding1
        self.equiv_class2.embedding = self.mock_embedding2
        self.equiv_class3.embedding = self.mock_embedding3
        
        # Create query embedding
        query_embedding = torch.tensor([0.9, 0.1, 0.0])
        
        # Get most similar nodes
        similar_nodes = self.plugin.get_most_similar_nodes(
            self.concept_graph, query_embedding, k=2
        )
        
        assert len(similar_nodes) <= 2
        for node_id, similarity in similar_nodes:
            assert node_id in ["equiv1", "equiv2", "equiv3"]
            assert 0.0 <= similarity <= 1.0
    
    def test_convenience_functions(self):
        """Test the convenience functions."""
        # Mock embeddings
        self.equiv_class1.embedding = self.mock_embedding1
        self.equiv_class2.embedding = self.mock_embedding2
        self.equiv_class3.embedding = self.mock_embedding3
        
        # Test compute_cosine_similarity_matrix
        similarity_matrix = compute_cosine_similarity_matrix(
            self.concept_graph, alpha=0.5, normalize=True
        )
        assert similarity_matrix.shape == (3, 3)
        
        # Test compute_similarity_to_query
        query_embedding = torch.tensor([0.9, 0.1, 0.0])
        similarities = compute_similarity_to_query(
            self.concept_graph, query_embedding, alpha=0.5, normalize=True
        )
        assert similarities.shape == (3,)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with empty graph
        empty_graph = ConceptGraph(nodes=[], edges=[])
        
        with pytest.raises(ValueError, match="No valid embeddings found"):
            self.plugin.compute_pairwise_similarity(empty_graph)
        
        # Test with invalid node IDs
        with pytest.raises(ValueError, match="No valid embeddings found"):
            self.plugin.compute_pairwise_similarity(
                self.concept_graph, node_ids=["nonexistent"]
            )


class TestCosineSimilarityIntegration:
    """Integration tests for cosine similarity with real concept graph scenarios."""
    
    def test_semantic_similarity_scenario(self):
        """Test a realistic scenario with semantically related concepts."""
        # Create concepts that should be semantically similar
        animal_concept = Concept(
            nodeid="animal",
            name="Animal",
            ground_set={"labels": ["Animal", "Creature", "Fauna"]}
        )
        
        mammal_concept = Concept(
            nodeid="mammal", 
            name="Mammal",
            ground_set={"labels": ["Mammal", "Warm-blooded animal"]}
        )
        
        dog_concept = Concept(
            nodeid="dog",
            name="Dog", 
            ground_set={"labels": ["Dog", "Canine", "Pet"]}
        )
        
        # Create equivalent classes
        animal_equiv = EquivalentClass([animal_concept])
        animal_equiv.id = "animal_equiv"
        
        mammal_equiv = EquivalentClass([mammal_concept])
        mammal_equiv.id = "mammal_equiv"
        
        dog_equiv = EquivalentClass([dog_concept])
        dog_equiv.id = "dog_equiv"
        
        # Create concept graph
        graph = ConceptGraph(
            nodes=[animal_equiv, mammal_equiv, dog_equiv],
            edges=[]
        )
        
        # Mock embeddings that reflect semantic relationships
        # Animal and Mammal should be more similar than Animal and Dog
        animal_equiv.embedding = torch.tensor([1.0, 0.0, 0.0, 0.0])
        mammal_equiv.embedding = torch.tensor([0.8, 0.6, 0.0, 0.0])  # Similar to animal
        dog_equiv.embedding = torch.tensor([0.0, 0.0, 1.0, 0.0])    # Different from animal
        
        plugin = CosineSimilarityPlugin(normalize=True)
        
        # Test pairwise similarities
        similarity_matrix = plugin.compute_pairwise_similarity(graph)
        
        # Animal-Mammal should be more similar than Animal-Dog
        animal_mammal_sim = similarity_matrix[0, 1]  # animal_equiv, mammal_equiv
        animal_dog_sim = similarity_matrix[0, 2]    # animal_equiv, dog_equiv
        
        assert animal_mammal_sim > animal_dog_sim
        
        # Test query similarity
        animal_query = torch.tensor([0.9, 0.1, 0.0, 0.0])
        similarities = plugin.compute_similarity_to_query(graph, animal_query)
        
        # Animal query should be most similar to animal_equiv
        assert similarities[0] > similarities[1]  # animal > mammal
        assert similarities[0] > similarities[2]  # animal > dog


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
