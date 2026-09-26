import pytest
from agentoi.algorithms.graph import (
    Concept,
    EquivalentClass,
    EquivalentClassRelation,
    ConceptGraph,
)
from agentoi.retrieval import NeighborhoodSampler, sample_neighborhood


def make_equiv_class(name: str) -> EquivalentClass:
    """wrap a single Concept into an EquivalentClass."""
    return EquivalentClass(equiv_concepts=[Concept(name=name)])

@pytest.fixture
def small_graph() -> ConceptGraph:
    # create four classes: A, B, C, D
    A = make_equiv_class("A")
    B = make_equiv_class("B")
    C = make_equiv_class("C")
    D = make_equiv_class("D")
    # edges: A->B, A->C, B->D
    edges = [
        EquivalentClassRelation(src=A, tgt=B, relation="yes", score=1.0),
        EquivalentClassRelation(src=A, tgt=C, relation="yes", score=1.0),
        EquivalentClassRelation(src=B, tgt=D, relation="yes", score=1.0),
    ]
    return ConceptGraph(nodes=[A, B, C, D], edges=edges)

def ids(equiv_list):
    """extract the set of ids from a list of EquivalentClass."""
    return {ec.id for ec in equiv_list}

def test_one_hop_neighbors(small_graph):
    # from B with max_hops=1: expect parents (A), children (D), and siblings (C)
    B = small_graph.nodes["B"]
    sampled = sample_neighborhood(small_graph, center=B, max_hops=1)
    assert ids(sampled) == {"A", "C", "D"}

def test_two_hop_neighbors(small_graph):
    # from B with max_hops=2: same as one hop in this small graph
    B = small_graph.nodes["B"]
    sampled = sample_neighborhood(small_graph, center=B, max_hops=2)
    assert ids(sampled) == {"A", "C", "D"}

def test_two_hop_from_root(small_graph):
    # from A with max_hops=2: 
    # 1-hop => B, C
    # 2-hop => children of B and C => D (via B) but none via C
    A = small_graph.nodes["A"]
    sampled = sample_neighborhood(small_graph, center=A, max_hops=2)
    assert ids(sampled) == {"B", "C", "D"}


def test_sampler_validates_center_and_hops(small_graph):
    unknown = make_equiv_class("missing")
    with pytest.raises(KeyError):
        sample_neighborhood(small_graph, unknown)
    with pytest.raises(ValueError):
        NeighborhoodSampler(max_hops=-1).sample(
            small_graph, small_graph.nodes["A"]
        )