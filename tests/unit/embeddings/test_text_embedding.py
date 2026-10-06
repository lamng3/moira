import torch
import pytest
from moira.embeddings import TextEmbedding
from moira.algorithms.graph import Concept, EquivalentClass


@pytest.fixture
def embedder():
    # singleton for all tests in this module
    return TextEmbedding()

def test_to_embedding_returns_tensor(embedder):
    emb = embedder.to_embedding("Hello, world!")
    # should be a 1‑D float tensor
    assert isinstance(emb, torch.Tensor)
    assert emb.ndim == 1
    assert emb.dtype == torch.float32 or emb.dtype == torch.float64

def test_embedding_has_nonzero_length(embedder):
    emb = embedder.to_embedding("Quick test")
    # dimensionality should be > 0
    assert emb.numel() > 0

def test_no_nans_or_infs(embedder):
    emb = embedder.to_embedding("Check for NaNs and Infs")
    assert not torch.isnan(emb).any(), "Embedding contains NaNs"
    assert not torch.isinf(emb).any(), "Embedding contains Infs"

def test_repeatability(embedder):
    text = "Repeatability"
    e1 = embedder.to_embedding(text)
    e2 = embedder.to_embedding(text)
    # identical calls should produce (numerically) the same output
    assert torch.allclose(e1, e2, atol=1e-6), "Embeddings differ between runs"

def test_different_texts_produce_different_embeddings(embedder):
    e1 = embedder.to_embedding("First sentence")
    e2 = embedder.to_embedding("A completely different sentence")
    # they shouldn’t be exactly equal
    assert not torch.allclose(e1, e2, atol=1e-6), "Different inputs gave identical embeddings"

def test_to_embedding_basic(embedder):
    emb = embedder.to_embedding("Hello, world!")
    assert isinstance(emb, torch.Tensor)
    assert emb.ndim == 1
    assert emb.numel() > 0
    assert not torch.isnan(emb).any()
    assert not torch.isinf(emb).any()

def test_compute_embedding_caches_on_node(embedder):
    c = Concept(name="Test", ground_set={"labels": ["w", "x"], "alt_labels": ["y", "z"]})
    eq = EquivalentClass([c])
    # first call computes and caches
    e1 = embedder.compute_embedding(eq)
    assert hasattr(eq, "text_embedding")
    # second call returns the same tensor object
    e2 = embedder.compute_embedding(eq)
    assert e2 is eq.text_embedding
    assert torch.allclose(e1, e2, atol=1e-6)

def test_ground_set_changes_embedding(embedder):
    # same name but different ground_set => different embeddings
    c1 = Concept(name="C", ground_set={"labels": ["a"]})
    c2 = Concept(name="C", ground_set={"alt_labels": ["b"]})
    eq1 = EquivalentClass([c1])
    eq2 = EquivalentClass([c2])
    e1 = embedder.compute_embedding(eq1)
    e2 = embedder.compute_embedding(eq2)
    assert e1.shape == e2.shape
    assert not torch.allclose(e1, e2, atol=1e-6)

def test_multiple_equiv_concepts(embedder):
    c1 = Concept(name="X", ground_set={"labels": ["a"]})
    c2 = Concept(name="Y", ground_set={"alt_labels": ["b"]})
    eq = EquivalentClass([c1, c2])
    emb = embedder.compute_embedding(eq)
    assert isinstance(emb, torch.Tensor)
    assert emb.ndim == 1
    # should not be zero vector
    zero = torch.zeros_like(emb)
    assert not torch.allclose(emb, zero)

def test_consistency_with_to_embedding(embedder):
    # when ground_set is empty, compute_embedding(eq) == to_embedding(name)
    c = Concept(name="Solo", ground_set={})
    eq = EquivalentClass([c])
    raw = embedder.to_embedding("Solo")
    via_eq = embedder.compute_embedding(eq)
    assert raw.shape == via_eq.shape
    assert torch.allclose(raw, via_eq, atol=1e-6)