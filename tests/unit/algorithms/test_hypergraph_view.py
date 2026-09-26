import pytest

from agentoi.algorithms.graph import Concept, EquivalentClass
from agentoi.algorithms.views import HypergraphView


class DummyGraph:
    def __init__(self, nodes):
        self.nodes = nodes


@pytest.fixture
def concepts_and_eq():
    # create some dummy Concept instances
    c1 = Concept(name="C1")
    c2 = Concept(name="C2")
    c3 = Concept(name="C3")
    # wrap them in EquivalentClass
    eq1 = EquivalentClass(equiv_concepts=[c1, c2])
    eq1.id = "eq1"
    eq2 = EquivalentClass(equiv_concepts=[c2, c3])
    eq2.id = "eq2"
    return c1, c2, c3, eq1, eq2


def test_initial_registration(concepts_and_eq):
    c1, c2, c3, eq1, eq2 = concepts_and_eq
    # graph with two equivalence classes
    graph = DummyGraph(nodes={"eq1": eq1, "eq2": eq2})
    hg = HypergraphView(graph)
    # both hyperedges should be registered
    assert set(hg.hyperedges.keys()) == {"eq1", "eq2"}
    # nodes_of should return correct member sets
    assert hg.nodes_of("eq1") == {c1.nodeid, c2.nodeid}
    assert hg.nodes_of("eq2") == {c2.nodeid, c3.nodeid}


def test_incidence_and_degree(concepts_and_eq):
    c1, c2, c3, eq1, eq2 = concepts_and_eq
    graph = DummyGraph(nodes={"eq1": eq1, "eq2": eq2})
    hg = HypergraphView(graph)
    # c2 belongs to both eq1 and eq2
    hyperedges_c2 = set(hg.hyperedges_of(c2))
    assert hyperedges_c2 == {"eq1", "eq2"}
    # degree should match number of incident hyperedges
    assert hg.degree(c2) == 2
    assert hg.degree(c1) == 1
    assert hg.degree(c3) == 1


def test_add_and_remove_hyperedge(concepts_and_eq):
    c1, _, c3, eq1, _ = concepts_and_eq
    # only eq1 initially registered (c1, c2)
    graph = DummyGraph(nodes={"eq1": eq1})
    hg = HypergraphView(graph)
    # add a new hyperedge for c1 and c3
    hg.add_hyperedge(edge_id="custom", members={c1.nodeid, c3.nodeid})
    assert "custom" in hg.hyperedges
    assert hg.nodes_of("custom") == {c1.nodeid, c3.nodeid}
    # only 'custom' should be registered for c3
    assert set(hg.hyperedges_of(c3)) == {"custom"}
    # remove existing hyperedge
    removed = hg.remove_hyperedge("custom")
    assert removed is True
    assert "custom" not in hg.hyperedges
    # c3 should have no hyperedges now
    assert hg.degree(c3) == 0
    # attempt to remove non-existent hyperedge
    assert hg.remove_hyperedge("nonexistent") is False


def test_hyperedges_of_returns_empty_for_unknown(concepts_and_eq):
    _, _, _, eq1, _ = concepts_and_eq
    graph = DummyGraph(nodes={"eq1": eq1})
    hg = HypergraphView(graph)
    # a concept not in any hyperedge should return empty list
    c_not = Concept(name="X")
    assert hg.hyperedges_of(c_not) == []
    assert hg.degree(c_not) == 0


if __name__ == "__main__":
    pytest.main()
