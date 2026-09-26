"""Canonical ontology predicate handling."""

from __future__ import annotations

from typing import Final

HIERARCHY_PREDICATES: Final[frozenset[str]] = frozenset(
    {"subclassof", "broader", "is_a", "isa", "is-a"}
)
EQUIVALENCE_PREDICATES: Final[frozenset[str]] = frozenset(
    {"equivalentclass", "sameas", "exactmatch", "equivalentto"}
)


def canonicalize_predicate(predicate: str | None) -> str:
    """Return a lowercase local name for an IRI, CURIE, or bare predicate."""
    value = (predicate or "").strip().strip("<>").lower()
    if "#" in value or "/" in value:
        separator = "#" if value.rfind("#") > value.rfind("/") else "/"
        value = value.rsplit(separator, 1)[-1]
    if ":" in value:
        value = value.split(":", 1)[1]
    return value


def is_hierarchy_predicate(predicate: str | None) -> bool:
    key = canonicalize_predicate(predicate)
    return key in HIERARCHY_PREDICATES or any(
        token in key for token in ("subclassof", "broader", "isa", "is-a")
    )


def is_equivalence_predicate(predicate: str | None) -> bool:
    return canonicalize_predicate(predicate) in EQUIVALENCE_PREDICATES
