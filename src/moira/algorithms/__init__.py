"""Graph models and ontology refinement algorithms."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "Concept": ("graph", "Concept"),
    "ConceptGraph": ("graph", "ConceptGraph"),
    "ConceptHypergraph": ("graph", "ConceptHypergraph"),
    "Ontology": ("graph", "Ontology"),
    "OfflineGraphRefiner": ("refinement", "OfflineGraphRefiner"),
    "OntologyRefiner": ("refinement", "OntologyRefiner"),
    "RelationRefiner": ("refinement", "RelationRefiner"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, symbol = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(f"{__name__}.{module_name}"), symbol)
    globals()[name] = value
    return value
