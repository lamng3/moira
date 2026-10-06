"""Graph data structures.

Public symbols are loaded on demand so importing lightweight graph models does
not import the optional embedding implementations.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "Concept": "concept",
    "ConceptRelation": "concept",
    "Ontology": "ontology",
    "EquivalentClass": "equivalence",
    "EquivalentClassRelation": "equivalence",
    "DSU": "equivalence",
    "ConceptGraph": "concept_graph",
    "ConceptHyperedge": "hypergraph",
    "ConceptHypergraph": "hypergraph",
    "OntologyStructureBuilder": "builder",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value
