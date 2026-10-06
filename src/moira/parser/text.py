"""Adapter for text relation-reasoning type vocabularies."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from moira.algorithms.graph import Ontology
from moira.algorithms.graph.identifiers import name_from_uuid

from .datasets import safe_uri_component
from .io import read_text
from .namespaces.constants import NAMESPACES
from .namespaces.registry import NamespaceRegistry


class TextParser:
    def __init__(
        self,
        name: str | None = None,
        version: str | None = None,
        namespaces: dict[str, Any] | None = None,
    ) -> None:
        self.lines: list[str] = []
        self.name = name or name_from_uuid(uuid.uuid4(), "codename")
        self.version = version or "0.1.0"
        self.namespaces = NamespaceRegistry()
        self.add_namespaces(NAMESPACES)
        if namespaces:
            self.add_namespaces(namespaces)

    def add_namespaces(self, mapping: dict[str, Any]) -> "TextParser":
        self.namespaces.update(mapping)
        return self

    def clear(self) -> "TextParser":
        self.lines = []
        return self

    def parse_text(self, text: str) -> "TextParser":
        self.lines = [line.strip() for line in text.splitlines() if line.strip()]
        return self

    def parse_file(self, path: str | Path) -> "TextParser":
        return self.parse_text(read_text(path))

    def extract_triples(self) -> list[tuple[str, str, str]]:
        triples: list[tuple[str, str, str]] = []
        for line in self.lines:
            type_uri = f"types:{safe_uri_component(line)}"
            triples.extend(
                [
                    (type_uri, "rdf:type", "kroma:TermType"),
                    (type_uri, "rdf:type", "owl:NamedIndividual"),
                    (type_uri, "rdfs:label", f'"{line}"'),
                    (type_uri, "skos:prefLabel", f'"{line}"'),
                    (type_uri, "rdfs:comment", f'"Term type: {line}"'),
                    (type_uri, "rdf:type", "kroma:VocabularyTerm"),
                ]
            )
        return triples

    def extract_node_info(self, node_id: str) -> dict[str, Any]:
        for line in self.lines:
            if f"types:{safe_uri_component(line)}" == node_id:
                return {
                    "n3": node_id,
                    "labels": [line],
                    "data_properties": {
                        "rdfs:label": [line],
                        "skos:prefLabel": [line],
                    },
                    "definitions": [],
                }
        return {"n3": node_id, "labels": [], "data_properties": {}, "definitions": []}

    def to_ontology(self) -> Ontology:
        ontology = Ontology(name=self.name, version=self.version)
        ontology = ontology.build_ontology_from_triples(self.extract_triples())
        ontology.metadata = {
            "source": "Text relation_reasoning dataset",
            "parser": type(self).__name__,
            "version": self.version,
            "name": self.name,
        }
        return ontology
