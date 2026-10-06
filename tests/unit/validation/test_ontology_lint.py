from rdflib import Graph, Literal, Namespace, OWL, RDF, RDFS, SKOS

from moira.validation.ontology_lint import exit_code, lint_graph


EX = Namespace("https://example.org/")


def valid_graph() -> Graph:
    graph = Graph()
    for item, label in ((EX.A, "A"), (EX.B, "B")):
        graph.add((item, RDF.type, OWL.Class))
        graph.add((item, RDFS.label, Literal(label)))
        graph.add((item, SKOS.altLabel, Literal(f"{label} alt")))
        graph.add((item, SKOS.definition, Literal(f"{label} definition")))
    graph.add((EX.A, RDFS.subClassOf, EX.B))
    graph.add((EX.related, RDF.type, OWL.ObjectProperty))
    graph.add((EX.related, RDFS.domain, EX.A))
    graph.add((EX.related, RDFS.range, EX.B))
    return graph


def test_valid_graph_passes() -> None:
    results = lint_graph(valid_graph(), check_endpoints=False)

    assert {result.status for result in results} == {"PASS"}
    assert exit_code(results) == 0


def test_missing_property_schema_fails() -> None:
    graph = valid_graph()
    graph.add((EX.unspecified, RDF.type, OWL.ObjectProperty))

    results = lint_graph(graph, check_endpoints=False)

    assert next(
        result for result in results if result.check == "missing_domain_range"
    ).status == "FAIL"
    assert exit_code(results) == 2
