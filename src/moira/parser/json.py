"""Adapter for JSON relation-reasoning datasets."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from moira.algorithms.graph import Ontology
from moira.algorithms.graph.identifiers import name_from_uuid

from .datasets import safe_uri_component
from .errors import ParseError
from .io import read_text
from .namespaces.constants import NAMESPACES
from .namespaces.registry import NamespaceRegistry


class JSONParser:
    def __init__(
        self,
        name: str | None = None,
        version: str | None = None,
        namespaces: dict[str, Any] | None = None,
    ) -> None:
        self.data: list[dict[str, Any]] = []
        self.name = name or name_from_uuid(uuid.uuid4(), "codename")
        self.version = version or "0.1.0"
        self.namespaces = NamespaceRegistry()
        self.add_namespaces(NAMESPACES)
        if namespaces:
            self.add_namespaces(namespaces)

    def add_namespaces(self, mapping: dict[str, Any]) -> "JSONParser":
        self.namespaces.update(mapping)
        return self

    def clear(self) -> "JSONParser":
        self.data = []
        return self

    def parse_text(
        self, text: str, *, source: str | Path | None = None
    ) -> "JSONParser":
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ParseError(
                f"Invalid JSON at line {exc.lineno}, column {exc.colno}",
                source=source,
                format="json",
            ) from exc
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise ParseError(
                "JSON dataset must be a list of objects", source=source, format="json"
            )
        self.data = value
        return self

    def parse_file(self, path: str | Path) -> "JSONParser":
        return self.parse_text(read_text(path), source=path)

    def extract_triples(self) -> list[tuple[str, str, str]]:
        triples: list[tuple[str, str, str]] = []
        for item in self.data:
            if "ID" not in item:
                continue
            term_id = str(item["ID"])
            term_uri = f"terms:{term_id}"
            triples.extend(
                [
                    (term_uri, "rdf:type", "kroma:Term"),
                    (term_uri, "rdf:type", "owl:NamedIndividual"),
                ]
            )
            if "term" in item:
                term = str(item["term"])
                triples.extend(
                    [
                        (term_uri, "kroma:hasText", f'"{term}"'),
                        (term_uri, "skos:prefLabel", f'"{term}"'),
                    ]
                )
            types = item.get("type", [])
            if not isinstance(types, list):
                types = [types]
            for type_name in types:
                label = str(type_name)
                type_uri = f"types:{safe_uri_component(label)}"
                triples.extend(
                    [
                        (term_uri, "kroma:hasType", type_uri),
                        (type_uri, "rdf:type", "kroma:TermType"),
                        (type_uri, "rdfs:label", f'"{label}"'),
                        (type_uri, "skos:prefLabel", f'"{label}"'),
                    ]
                )
            if item.get("sentence"):
                triples.append(
                    (term_uri, "kroma:hasContext", f'"{item["sentence"]}"')
                )
        return triples

    def extract_node_info(self, node_id: str) -> dict[str, Any]:
        raw_id = node_id.removeprefix("terms:")
        for item in self.data:
            if str(item.get("ID")) == raw_id:
                term = str(item["term"]) if "term" in item else ""
                context = str(item["sentence"]) if item.get("sentence") else ""
                return {
                    "n3": f"terms:{raw_id}",
                    "labels": [term] if term else [],
                    "data_properties": {
                        "kroma:hasText": [term] if term else [],
                        "kroma:hasContext": [context] if context else [],
                    },
                    "definitions": [],
                }
        return {"n3": node_id, "labels": [], "data_properties": {}, "definitions": []}

    def to_ontology(self) -> Ontology:
        ontology = Ontology(name=self.name, version=self.version)
        ontology = ontology.build_ontology_from_triples(self.extract_triples())
        ontology.metadata = {
            "source": "JSON relation_reasoning dataset",
            "parser": type(self).__name__,
            "version": self.version,
            "name": self.name,
        }
        return ontology
