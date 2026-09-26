import subprocess
import sys

import pytest
import torch

from agentoi.embeddings import (
    MultiQuerySearchService,
    TorchExactVectorIndex,
    create_vector_index,
    fuse_embeddings,
)
from agentoi.embeddings.vector_index import VectorIndexRegistry


def test_public_packages_are_lazy():
    code = """
import sys
sys.path.insert(0, "src")
import agentoi.embeddings
import agentoi.algorithms.graph
assert "torch" not in sys.modules
assert "transformers" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_fusion_uses_graph_alpha_and_validates_shapes():
    graph = torch.tensor([2.0, 4.0])
    text = torch.tensor([6.0, 8.0])

    assert torch.equal(
        fuse_embeddings(graph, text, alpha=0.25),
        torch.tensor([5.0, 7.0]),
    )
    with pytest.raises(ValueError, match="dimensions"):
        fuse_embeddings(graph, torch.tensor([1.0]), alpha=0.5)


def test_exact_cosine_search_metadata_upsert_and_remove():
    index = TorchExactVectorIndex(dimension=2)
    index.add(
        ["east", "north", "west"],
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
        [{"direction": "e"}, None, {"direction": "w"}],
    )

    results = index.search(torch.tensor([1.0, 0.0]), k=2)
    assert [result.id for result in results] == ["east", "north"]
    assert results[0].score == pytest.approx(1.0)
    assert results[0].metadata == {"direction": "e"}

    index.add(["north"], torch.tensor([[1.0, 0.0]]))
    assert index.search(torch.tensor([1.0, 0.0]), k=2)[1].id == "north"
    index.remove(["east", "missing"])
    assert len(index) == 2


def test_factory_is_extensible_without_loading_optional_backends():
    index = create_vector_index("torch-exact", dimension=3)
    assert isinstance(index, TorchExactVectorIndex)

    registry = VectorIndexRegistry()
    sentinel = object()
    registry.register("remote", lambda **kwargs: kwargs["instance"])
    assert registry.create("REMOTE", instance=sentinel) is sentinel


def test_index_nodes_accepts_an_empty_collection():
    index = TorchExactVectorIndex(dimension=2)

    assert MultiQuerySearchService.index_nodes(index, [], clear=True) is index
    assert len(index) == 0
