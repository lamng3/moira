"""Ontology quality checks and command-line reporting."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import requests
from rdflib import Graph, OWL, RDF, RDFS, SKOS, URIRef


@dataclass(frozen=True)
class LintResult:
    """One ontology quality result."""

    check: str
    value: Any
    status: str


def _classes(graph: Graph) -> set:
    return set(graph.subjects(RDF.type, OWL.Class)) | set(
        graph.subjects(RDF.type, RDFS.Class)
    )


def isolated_classes(graph: Graph) -> list[str]:
    """Return classes outside the subclass graph."""
    classes = _classes(graph)
    connected = set(graph.subjects(RDFS.subClassOf, None)) | set(
        graph.objects(None, RDFS.subClassOf)
    )
    return sorted(str(item) for item in classes - connected)


def connected_components(graph: Graph) -> int:
    """Count components in the class hierarchy."""
    hierarchy = nx.Graph()
    hierarchy.add_nodes_from(str(item) for item in _classes(graph))
    hierarchy.add_edges_from(
        (str(child), str(parent))
        for child, parent in graph.subject_objects(RDFS.subClassOf)
    )
    return nx.number_connected_components(hierarchy) if hierarchy else 0


def properties_without_schema(graph: Graph) -> list[str]:
    """Return properties with neither domain nor range."""
    properties = (
        set(graph.subjects(RDF.type, RDF.Property))
        | set(graph.subjects(RDF.type, OWL.ObjectProperty))
        | set(graph.subjects(RDF.type, OWL.DatatypeProperty))
    )
    return sorted(
        str(prop)
        for prop in properties
        if not any(graph.objects(prop, RDFS.domain))
        and not any(graph.objects(prop, RDFS.range))
    )


def duplicate_labels(graph: Graph) -> dict[str, list[str]]:
    """Return labels assigned to multiple resources."""
    labels: dict[str, list[str]] = {}
    for subject, label in graph.subject_objects(RDFS.label):
        labels.setdefault(str(label), []).append(str(subject))
    return {label: subjects for label, subjects in labels.items() if len(subjects) > 1}


def skos_coverage(graph: Graph) -> tuple[float, float]:
    """Return alt-label and definition coverage percentages."""
    concepts = _classes(graph) | set(graph.subjects(RDF.type, SKOS.Concept))
    if not concepts:
        return 100.0, 100.0

    alt_labels = sum(any(graph.objects(item, SKOS.altLabel)) for item in concepts)
    definitions = sum(
        any(graph.objects(item, SKOS.definition))
        or any(graph.objects(item, RDFS.comment))
        for item in concepts
    )
    total = len(concepts)
    return round(100 * alt_labels / total, 1), round(100 * definitions / total, 1)


def unreachable_endpoints(graph: Graph, timeout: float = 5.0) -> list[str]:
    """Return unavailable VoID dumps and SPARQL endpoints."""
    predicates = (
        URIRef("http://rdfs.org/ns/void#dataDump"),
        URIRef("http://rdfs.org/ns/void#sparqlEndpoint"),
    )
    urls = {str(url) for predicate in predicates for url in graph.objects(None, predicate)}
    unavailable = []
    for url in sorted(urls):
        try:
            if requests.head(url, timeout=timeout, allow_redirects=True).status_code >= 400:
                unavailable.append(url)
        except requests.RequestException:
            unavailable.append(url)
    return unavailable


def lint_graph(graph: Graph, check_endpoints: bool = True) -> list[LintResult]:
    """Run MOIRA ontology quality checks."""
    isolated = isolated_classes(graph)
    components = connected_components(graph)
    missing_schema = properties_without_schema(graph)
    duplicates = duplicate_labels(graph)
    alt_labels, definitions = skos_coverage(graph)

    results = [
        LintResult("isolated_classes", len(isolated), "WARN" if isolated else "PASS"),
        LintResult(
            "connected_components",
            components,
            "WARN" if components > 1 else "PASS",
        ),
        LintResult(
            "missing_domain_range",
            len(missing_schema),
            "FAIL" if missing_schema else "PASS",
        ),
        LintResult(
            "duplicate_labels",
            len(duplicates),
            "WARN" if duplicates else "PASS",
        ),
        LintResult(
            "skos_alt_label_coverage",
            f"{alt_labels}%",
            "WARN" if alt_labels < 60 else "PASS",
        ),
        LintResult(
            "definition_coverage",
            f"{definitions}%",
            "WARN" if definitions < 40 else "PASS",
        ),
    ]

    if check_endpoints:
        unavailable = unreachable_endpoints(graph)
        results.append(
            LintResult(
                "endpoint_reachability",
                len(unavailable),
                "FAIL" if unavailable else "PASS",
            )
        )
    return results


def write_report(results: Iterable[LintResult], path: Path) -> None:
    """Write lint results as CSV."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Check", "Result", "Status"])
        writer.writerows((item.check, item.value, item.status) for item in results)


def exit_code(results: Iterable[LintResult]) -> int:
    """Map lint statuses to a process exit code."""
    statuses = {item.status for item in results}
    if "FAIL" in statuses:
        return 2
    return 1 if "WARN" in statuses else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check ontology quality.")
    parser.add_argument("ontology", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--skip-endpoints", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.ontology.is_file():
        raise SystemExit(f"Ontology not found: {args.ontology}")

    graph = Graph()
    graph.parse(args.ontology)
    results = lint_graph(graph, check_endpoints=not args.skip_endpoints)
    output = args.output or args.ontology.with_suffix(".lint.csv")
    write_report(results, output)

    for item in results:
        print(f"{item.status:4}  {item.check:28} {item.value}")
    print(f"Report: {output}")
    return exit_code(results)


if __name__ == "__main__":
    raise SystemExit(main())
