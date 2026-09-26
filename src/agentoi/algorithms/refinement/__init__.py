"""Ontology relation and graph refinement."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .offline import OfflineGraphRefiner
    from .online import (
        DeferredReason,
        DeferredUpdate,
        GraphUpdate,
        OnlineOntologyRefiner,
        OnlineRefinementResult,
    )
    from .ontology import OntologyRefiner
    from .pipeline import (
        RefinementDiagnostic,
        RefinementOptions,
        RefinementResult,
        RelationRefiner,
        refine_relations,
    )
    from .rules import RuleSet, Triple, default_rules, load_rules, load_rules_from_json

_EXPORTS = {
    "OfflineGraphRefiner": "offline",
    "OnlineOntologyRefiner": "online",
    "OnlineRefinementResult": "online",
    "GraphUpdate": "online",
    "DeferredUpdate": "online",
    "DeferredReason": "online",
    "OntologyRefiner": "ontology",
    "RefinementDiagnostic": "pipeline",
    "RefinementOptions": "pipeline",
    "RefinementResult": "pipeline",
    "RelationRefiner": "pipeline",
    "refine_relations": "pipeline",
    "RuleSet": "rules",
    "Triple": "rules",
    "default_rules": "rules",
    "load_rules": "rules",
    "load_rules_from_json": "rules",
}


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


__all__ = [
    "OfflineGraphRefiner",
    "OnlineOntologyRefiner",
    "OnlineRefinementResult",
    "GraphUpdate",
    "DeferredUpdate",
    "DeferredReason",
    "OntologyRefiner",
    "RefinementDiagnostic",
    "RefinementOptions",
    "RefinementResult",
    "RelationRefiner",
    "RuleSet",
    "Triple",
    "default_rules",
    "load_rules",
    "load_rules_from_json",
    "refine_relations",
]
