"""Path-oriented facade that selects a focused parser adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rdflib.namespace import OWL, RDF

from .config import ParserConfig
from .errors import ParserConfigError
from .formats import InputFormat, parse_format
from .json import JSONParser
from .rdf import RDFParser
from .text import TextParser

ParserAdapter = RDFParser | JSONParser | TextParser


class Parser:
    """Parse a local ontology or relation-reasoning dataset."""

    def __init__(
        self,
        filepath: str | Path,
        config_path: str | Path | None = None,
        name: str | None = None,
        version: str | None = None,
        *,
        config: ParserConfig | None = None,
        format: str | InputFormat | None = None,
    ) -> None:
        if config is not None and config_path is not None:
            raise ParserConfigError("Pass either config or config_path, not both")
        self.path = Path(filepath)
        self.config = config or (
            ParserConfig.from_file(config_path) if config_path else ParserConfig()
        )
        self.name = name
        self.version = version
        self._requested_format = parse_format(format) if format else None
        self._parser: ParserAdapter | None = None
        self._format: InputFormat | None = None

    def _ensure_parser(self) -> None:
        if self._parser is not None:
            return

        selection = self.config.select(self.path, self._requested_format)
        selected_format = self._requested_format or selection.format
        parser_name = selection.name or self.name
        parser_version = selection.version or self.version
        arguments: dict[str, Any] = {
            "name": parser_name,
            "version": parser_version,
            "namespaces": selection.namespaces,
        }

        if selected_format.adapter == "rdf":
            backend: ParserAdapter = RDFParser(**arguments)
            backend.parse_file(self.path, format=selected_format)
            self._bind_ontology_namespace(backend)
        elif selected_format is InputFormat.JSON:
            backend = JSONParser(**arguments)
            backend.parse_file(self.path)
        else:
            backend = TextParser(**arguments)
            backend.parse_file(self.path)

        self._format = selected_format
        self._parser = backend

    def _bind_ontology_namespace(self, backend: RDFParser) -> None:
        existing = {prefix for prefix, _ in backend.graph.namespaces()}
        stem = _uncompressed_stem(self.path)
        if stem in existing:
            return
        base_iri = next(
            (str(subject) for subject in backend.graph.subjects(RDF.type, OWL.Ontology)),
            None,
        )
        if base_iri:
            namespace = (
                base_iri if base_iri.endswith(("#", "/", ":")) else f"{base_iri}#"
            )
            backend.add_namespaces({stem: namespace, f"{stem}base": base_iri})

    @property
    def format(self) -> str:
        self._ensure_parser()
        assert self._format is not None
        return self._format.value

    @property
    def parser(self) -> ParserAdapter:
        self._ensure_parser()
        assert self._parser is not None
        return self._parser

    def to_ontology(self):
        return self.parser.to_ontology()

    def extract_triples(self, *args: Any, **kwargs: Any):
        return self.parser.extract_triples(*args, **kwargs)

    def extract_node_info(self, subject: str):
        return self.parser.extract_node_info(subject)

    def qname(self, node: str) -> str:
        if not isinstance(self.parser, RDFParser):
            raise TypeError("qname is only available for RDF inputs")
        return self.parser.qname(node)


def _uncompressed_stem(path: Path) -> str:
    if path.suffix.lower() in {".gz", ".bz2"}:
        return path.with_suffix("").stem
    return path.stem
