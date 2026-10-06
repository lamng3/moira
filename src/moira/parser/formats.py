"""Input format detection and parser adapter selection."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from .errors import UnsupportedFormatError


class InputFormat(str, Enum):
    TURTLE = "turtle"
    N3 = "n3"
    NTRIPLES = "nt"
    TRIG = "trig"
    TRIX = "trix"
    RDF_XML = "xml"
    JSON_LD = "json-ld"
    JSON = "json"
    TEXT = "text"

    @property
    def adapter(self) -> str:
        if self in {InputFormat.JSON, InputFormat.TEXT}:
            return self.value
        return "rdf"


_EXTENSIONS = {
    ".ttl": InputFormat.TURTLE,
    ".n3": InputFormat.N3,
    ".nt": InputFormat.NTRIPLES,
    ".trig": InputFormat.TRIG,
    ".trix": InputFormat.TRIX,
    ".owl": InputFormat.RDF_XML,
    ".rdf": InputFormat.RDF_XML,
    ".xml": InputFormat.RDF_XML,
    ".jsonld": InputFormat.JSON_LD,
    ".json": InputFormat.JSON,
    ".txt": InputFormat.TEXT,
}

_ALIASES = {
    "ttl": InputFormat.TURTLE,
    "application/rdf+xml": InputFormat.RDF_XML,
    "rdf/xml": InputFormat.RDF_XML,
    "rdf-xml": InputFormat.RDF_XML,
    "application/ld+json": InputFormat.JSON_LD,
    "jsonld": InputFormat.JSON_LD,
}

COMPRESSION_SUFFIXES = {".gz", ".bz2"}


def parse_format(value: str | InputFormat) -> InputFormat:
    """Normalize a configured format name."""
    if isinstance(value, InputFormat):
        return value
    normalized = value.strip().lower()
    if normalized in _ALIASES:
        return _ALIASES[normalized]
    try:
        return InputFormat(normalized)
    except ValueError as exc:
        raise UnsupportedFormatError(f"Unsupported parser format: {value!r}") from exc


def detect_format(path: str | Path) -> InputFormat:
    """Detect an input format from its final non-compression suffix."""
    source = Path(path)
    suffixes = [suffix.lower() for suffix in source.suffixes]
    if suffixes and suffixes[-1] in COMPRESSION_SUFFIXES:
        suffixes.pop()
    if not suffixes or suffixes[-1] not in _EXTENSIONS:
        raise UnsupportedFormatError(f"Cannot detect parser format from {source}")
    return _EXTENSIONS[suffixes[-1]]
