import os
import subprocess
import sys

from agentoi.algorithms.graph import (
    Concept,
    ConceptGraph,
    ConceptHypergraph,
    EquivalentClass,
    Ontology,
)
from agentoi.algorithms.graph.predicates import (
    canonicalize_predicate,
    is_equivalence_predicate,
    is_hierarchy_predicate,
)
from agentoi.algorithms.views import HypergraphView, MultigraphView


def test_classes_are_owned_by_new_packages():
    assert Concept.__module__ == "agentoi.algorithms.graph.concept"
    assert Ontology.__module__ == "agentoi.algorithms.graph.ontology"
    assert EquivalentClass.__module__ == "agentoi.algorithms.graph.equivalence"
    assert ConceptGraph.__module__ == "agentoi.algorithms.graph.concept_graph"
    assert ConceptHypergraph.__module__ == "agentoi.algorithms.graph.hypergraph"
    assert HypergraphView.__module__ == "agentoi.algorithms.views.hypergraph"
    assert MultigraphView.__module__ == "agentoi.algorithms.views.multigraph"


def test_predicates_share_canonicalization():
    assert (
        canonicalize_predicate("http://www.w3.org/2000/01/rdf-schema#subClassOf")
        == "subclassof"
    )
    assert is_hierarchy_predicate("rdfs:subClassOf")
    assert is_equivalence_predicate("owl:equivalentClass")


def test_graph_and_hypergraph_share_equivalence_collapse():
    ontology = Ontology().build_ontology_from_triples(
        [("A", "owl:equivalentClass", "B"), ("A", "rdfs:subClassOf", "C")]
    )
    graph = ConceptGraph([], []).build_from_ontology(ontology)
    assert sorted(len(node.equiv_concepts) for node in graph.nodes.values()) == [1, 2]


def test_empty_hypergraph_view_has_hierarchy_maps():
    view = HypergraphView(ConceptGraph([], []))
    assert view.parents_of_edge("missing") == []
    assert view.children_of_edge("missing") == []


def test_graph_import_does_not_load_embedding_modules():
    code = (
        "import sys; from agentoi.algorithms.graph import Concept, Ontology; "
        "assert not any(name.startswith('agentoi.embeddings') for name in sys.modules)"
    )
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(("src", os.environ.get("PYTHONPATH", ""))),
    }
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
