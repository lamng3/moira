from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from agentoi.functions.similarity import (
    ComparisonResult,
    CosineSimilarityPlugin,
    NodePairSimilarityEngine,
    SimilarityComparisonSystem,
    SimilarityMethod,
)


def _graph(*node_ids: str) -> SimpleNamespace:
    return SimpleNamespace(
        nodes={node_id: SimpleNamespace(id=node_id) for node_id in node_ids}
    )


def _system(scores: dict[tuple[str, str], tuple[float, float]]) -> SimilarityComparisonSystem:
    system = SimilarityComparisonSystem()

    def cosine(node_1, node_2):
        return scores[tuple(sorted((node_1.id, node_2.id)))][0]

    def google(node_1, node_2):
        return scores[tuple(sorted((node_1.id, node_2.id)))][1]

    system.engine.cosine = Mock(side_effect=cosine)
    system.engine.google_index = Mock(side_effect=google)
    return system


def test_compute_similarity_hybrid_defaults_to_scalar_and_caches() -> None:
    graph = _graph("a", "b")
    system = _system({("a", "b"): (0.8, 0.2)})

    first = system.compute_similarity(graph, "a", "b", SimilarityMethod.HYBRID)
    second = system.compute_similarity(graph, "b", "a", SimilarityMethod.HYBRID)

    assert first == pytest.approx(0.5)
    assert second == pytest.approx(0.5)
    assert isinstance(first, float)
    assert system.engine.cosine.call_count == 1
    assert system.engine.google_index.call_count == 1


def test_hybrid_return_all_has_an_independent_cache_entry() -> None:
    graph = _graph("a", "b")
    system = _system({("a", "b"): (0.8, 0.2)})

    scalar = system.compute_similarity(
        graph, "a", "b", SimilarityMethod.HYBRID
    )
    all_scores = system.compute_similarity(
        graph, "a", "b", SimilarityMethod.HYBRID, return_all=True
    )
    cached = system.compute_similarity(
        graph, "b", "a", SimilarityMethod.HYBRID, return_all=True
    )

    assert scalar == pytest.approx(0.5)
    assert all_scores == pytest.approx((0.8, 0.2, 0.5))
    assert cached == all_scores
    assert system.engine.cosine.call_count == 2
    assert system.engine.google_index.call_count == 2


def test_compare_similarities_reuses_scores_without_sharing_mutable_results() -> None:
    graph = _graph("a", "b")
    system = _system({("a", "b"): (0.9, 0.3)})

    result = system.compare_similarities(graph, "a", "b")
    reversed_result = system.compare_similarities(graph, "b", "a")

    assert isinstance(result, ComparisonResult)
    assert result.cosine_similarity == pytest.approx(0.9)
    assert result.google_similarity == pytest.approx(0.3)
    assert result.hybrid_similarity == pytest.approx(0.6)
    assert reversed_result is not result
    assert reversed_result.hybrid_similarity == result.hybrid_similarity
    assert system.engine.cosine.call_count == 1
    assert system.engine.google_index.call_count == 1


def test_hybrid_top_k_uses_scalar_hybrid_score() -> None:
    graph = _graph("a", "b", "c")
    system = _system(
        {
            ("a", "b"): (0.9, 0.1),
            ("a", "c"): (0.4, 0.8),
            ("b", "c"): (0.2, 0.2),
        }
    )

    results = system.get_top_k_pairs(
        graph, k=2, method=SimilarityMethod.HYBRID
    )

    assert [(result.node_id_1, result.node_id_2) for result in results] == [
        ("a", "c"),
        ("a", "b"),
    ]
    assert all(isinstance(result.similarity_score, float) for result in results)
    assert [result.similarity_score for result in results] == pytest.approx([0.6, 0.5])


def test_compare_top_k_assigns_deterministic_method_ranks() -> None:
    graph = _graph("a", "b", "c")
    system = _system(
        {
            ("a", "b"): (0.9, 0.1),
            ("a", "c"): (0.4, 0.8),
            ("b", "c"): (0.2, 0.2),
        }
    )

    results = system.compare_top_k_pairs(graph, k=3)

    assert [(result.node_id_1, result.node_id_2) for result in results] == [
        ("a", "c"),
        ("a", "b"),
        ("b", "c"),
    ]
    assert [result.hybrid_rank for result in results] == [1, 2, 3]
    by_pair = {(result.node_id_1, result.node_id_2): result for result in results}
    assert by_pair[("a", "b")].cosine_rank == 1
    assert by_pair[("a", "c")].google_rank == 1


def test_cache_is_scoped_to_graph_identity() -> None:
    first_graph = _graph("a", "b")
    second_graph = _graph("a", "b")
    system = SimilarityComparisonSystem()
    system.engine.cosine = Mock(side_effect=[0.9, 0.1])

    first = system.compute_similarity(
        first_graph, "a", "b", SimilarityMethod.COSINE
    )
    second = system.compute_similarity(
        second_graph, "a", "b", SimilarityMethod.COSINE
    )

    assert (first, second) == (0.9, 0.1)
    assert system.engine.cosine.call_count == 2


def test_self_pairs_are_included_when_requested() -> None:
    graph = _graph("a", "b")
    system = SimilarityComparisonSystem()
    system.engine.cosine = Mock(return_value=1.0)

    results = system.get_top_k_pairs(
        graph,
        k=3,
        method=SimilarityMethod.COSINE,
        exclude_self=False,
    )

    assert [(item.node_id_1, item.node_id_2) for item in results] == [
        ("a", "a"),
        ("a", "b"),
        ("b", "b"),
    ]


@pytest.mark.parametrize("node_ids", [[], ["only"]])
def test_analysis_handles_fewer_than_two_nodes(node_ids) -> None:
    graph = _graph(*node_ids)
    result = SimilarityComparisonSystem().analyze_similarity_methods(graph)

    assert result["num_pairs"] == 0
    assert result["statistics"]["cosine"]["mean"] == 0.0


def test_engine_respects_disabled_normalization() -> None:
    first = SimpleNamespace(
        id="a", embedding=torch.tensor([0.2, 0.0]), equiv_concepts=[]
    )
    second = SimpleNamespace(
        id="b", embedding=torch.tensor([0.2, 0.0]), equiv_concepts=[]
    )
    engine = NodePairSimilarityEngine(
        cosine_plugin=CosineSimilarityPlugin(normalize=False)
    )

    assert engine.cosine(first, second) == pytest.approx(0.04)


def test_cache_tracks_replaced_embeddings() -> None:
    graph = _graph("a", "b")
    graph.nodes["a"].embedding = torch.tensor([1.0, 0.0])
    graph.nodes["b"].embedding = torch.tensor([1.0, 0.0])
    graph.nodes["a"].equiv_concepts = []
    graph.nodes["b"].equiv_concepts = []
    system = SimilarityComparisonSystem()

    first = system.compute_similarity(
        graph, "a", "b", SimilarityMethod.COSINE
    )
    graph.nodes["b"].embedding = torch.tensor([0.0, 1.0])
    second = system.compute_similarity(
        graph, "a", "b", SimilarityMethod.COSINE
    )

    assert first == pytest.approx(1.0)
    assert second == pytest.approx(0.0)
