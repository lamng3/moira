from types import SimpleNamespace

import pytest
import torch

from agentoi.functions.similarity import (
    EmbeddingAwareSimilaritySystem,
    create_embedding_aware_system,
)
from agentoi.retrieval.web import DatasetSearchProvider, StaticSearchProvider


def _node(node_id: str, values: list[float]):
    embedding = torch.tensor(values, requires_grad=True)
    concept = SimpleNamespace(
        name=node_id,
        ground_set={"labels": [node_id], "exact_synonyms": []},
    )
    return SimpleNamespace(
        id=node_id,
        equiv_concepts=[concept],
        embedding=embedding,
        text_embedding=embedding,
        graph_embedding=embedding,
    )


def _graph(*nodes):
    return SimpleNamespace(
        nodes={node.id: node for node in nodes},
        compute_all_embeddings=lambda alpha: None,
    )


def test_factory_defaults_to_seeded_offline_provider_and_exposes_embedding_flag():
    graph = _graph(_node("a", [1.0, 0.0]), _node("b", [0.0, 1.0]))

    first = create_embedding_aware_system(
        graph, provider_seed=17, ensure_embeddings=False
    )
    second = create_embedding_aware_system(
        graph, provider_seed=17, ensure_embeddings=False
    )

    assert isinstance(first.search_provider, DatasetSearchProvider)
    assert first.search_provider.counts == second.search_provider.counts
    assert first.ensure_embeddings is False


def test_explicit_provider_is_preserved_and_embedding_stats_detach_tensors():
    graph = _graph(_node("a", [3.0, 4.0]), _node("b", [0.0, 2.0]))
    provider = StaticSearchProvider({})
    system = EmbeddingAwareSimilaritySystem(graph, search_provider=provider)

    stats = system._analyze_embeddings()

    assert system.search_provider is provider
    assert stats["fused_embeddings"] == {
        "count": 2,
        "dimension": 2,
        "mean_norm": pytest.approx(3.5),
        "std_norm": pytest.approx(1.5),
    }


def test_embedding_initialization_failure_is_raised():
    node = SimpleNamespace(
        id="a",
        equiv_concepts=[],
        embedding=None,
        text_embedding=None,
        graph_embedding=None,
    )

    def fail(*, alpha):
        raise ValueError("encoder unavailable")

    graph = SimpleNamespace(nodes={"a": node}, compute_all_embeddings=fail)
    system = create_embedding_aware_system(graph)

    with pytest.raises(RuntimeError, match="Failed to compute graph embeddings"):
        system._ensure_embeddings_computed(["a"])
