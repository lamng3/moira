import pytest

from agentoi.algorithms.graph import Concept, EquivalentClass
from agentoi.algorithms.views import MultigraphView


class DummyGraph:
    def __init__(self, nodes: dict, edges: list | None = None):
        self.nodes = nodes
        self.edges = edges or []


@pytest.fixture
def graph_and_nodes():
    # create two dummy Concept instances wrapped in EquivalentClass
    c1 = Concept(name="C1")
    c2 = Concept(name="C2")
    eq1 = EquivalentClass(equiv_concepts=[c1])
    eq1.id = "n1"
    eq2 = EquivalentClass(equiv_concepts=[c2])
    eq2.id = "n2"
    # graph stub with nodes dictionary
    graph = DummyGraph(nodes={eq1.id: eq1, eq2.id: eq2})
    return graph, eq1, eq2


@pytest.fixture
def mv_and_nodes(graph_and_nodes):
    graph, n1, n2 = graph_and_nodes
    mv = MultigraphView(graph)
    # ensure MV has access to node ids
    mv.nodes = graph.nodes
    return mv, n1.id, n2.id


def test_empty_initialization(mv_and_nodes):
    mv, n1, n2 = mv_and_nodes
    # no edges initially
    assert mv.edges == []
    assert mv.adjacency[n1] == []
    assert mv.adjacency[n2] == []
    assert mv.reverse_adjacency[n1] == []
    assert mv.reverse_adjacency[n2] == []


def test_add_edge_and_incident(mv_and_nodes):
    mv, n1, n2 = mv_and_nodes
    # add a single edge from n1 to n2
    mv.add_edge(edge_id="e1", src_id=n1, tgt_id=n2, metadata={"weight": 0.5})
    # edge recorded
    assert mv.edges == [("e1", n1, n2, {"weight": 0.5})]
    # adjacency updated
    assert mv.adjacency[n1] == ["e1"]
    assert mv.reverse_adjacency[n2] == ["e1"]
    # edges_between
    assert mv.edges_between(n1, n2) == ["e1"]
    # incident_edges
    assert mv.incident_edges(n1) == ["e1"]
    assert mv.incident_edges(n2) == ["e1"]


def test_parallel_edges_and_loop(mv_and_nodes):
    mv, n1, n2 = mv_and_nodes
    # add parallel edges between n1 and n2
    mv.add_edge("e1", n1, n2)
    mv.add_edge("e2", n1, n2)
    assert set(mv.edges_between(n1, n2)) == {"e1", "e2"}
    # incident_edges for n1 should list both in order
    assert mv.incident_edges(n1) == ["e1", "e2"]
    # add a loop on n1
    mv.add_edge("loop", n1, n1)
    assert mv.edges_between(n1, n1) == ["loop"]
    # loop appears once in incident_edges
    assert mv.incident_edges(n1) == ["e1", "e2", "loop"]


def test_remove_edge(mv_and_nodes):
    mv, n1, n2 = mv_and_nodes
    # setup edges
    mv.add_edge("e1", n1, n2)
    mv.add_edge("e2", n1, n2)
    # remove e1
    removed = mv.remove_edge("e1")
    assert removed is True
    assert mv.edges_between(n1, n2) == ["e2"]
    assert mv.adjacency[n1] == ["e2"]
    assert mv.reverse_adjacency[n2] == ["e2"]
    # removing non-existent edge
    assert mv.remove_edge("nonexistent") is False


def test_initial_multiedges_parameter(graph_and_nodes):
    graph, n1, n2 = graph_and_nodes
    # initialize without multiedges then add edges manually
    mv = MultigraphView(graph)
    mv.nodes = graph.nodes
    initial = [("init", n1.id, n2.id, {"score": 1})]
    for eid, src, tgt, meta in initial:
        mv.add_edge(eid, src, tgt, meta)
    # verify the edge
    assert mv.edges_between(n1.id, n2.id) == ["init"]
    assert mv.incident_edges(n1.id) == ["init"]
    assert mv.incident_edges(n2.id) == ["init"]


if __name__ == "__main__":
    pytest.main()
