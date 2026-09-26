from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, TypeAlias

from agentoi.agents import Agent
from agentoi.algorithms.graph import Ontology


SimilarityFunction: TypeAlias = Callable[[Any, Any], float]


@dataclass(frozen=True, slots=True)
class OntologyMatchingConfiguration:
    """Validated ontology-matching configuration W = (Oₛ, Oₜ, S, M, τ)."""

    source_ontology: Ontology
    target_ontology: Ontology
    similarity_functions: tuple[SimilarityFunction, ...]
    models: tuple[Agent, ...]
    threshold: float

    def __post_init__(self) -> None:
        if not isinstance(self.source_ontology, Ontology):
            raise TypeError("source_ontology must be an Ontology")
        if not isinstance(self.target_ontology, Ontology):
            raise TypeError("target_ontology must be an Ontology")
        if self.source_ontology is self.target_ontology:
            raise ValueError("source and target ontologies must be distinct objects")
        if not self.similarity_functions:
            raise ValueError("at least one similarity function is required")
        if not all(callable(function) for function in self.similarity_functions):
            raise TypeError("similarity_functions must contain only callables")
        if not self.models:
            raise ValueError("at least one model is required")
        if not all(isinstance(model, Agent) for model in self.models):
            raise TypeError("models must contain only Agent instances")
        if isinstance(self.threshold, bool) or not isinstance(
            self.threshold, (int, float)
        ):
            raise TypeError("threshold must be a number")
        if not 0.0 <= float(self.threshold) <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        object.__setattr__(self, "threshold", float(self.threshold))


def build_configuration_w(
    source_ontology: Ontology,
    target_ontology: Ontology,
    similarity_functions: Iterable[SimilarityFunction],
    models: Iterable[Agent],
    threshold: float,
) -> OntologyMatchingConfiguration:
    """Build and validate ontology-matching configuration W."""
    return OntologyMatchingConfiguration(
        source_ontology=source_ontology,
        target_ontology=target_ontology,
        similarity_functions=tuple(similarity_functions),
        models=tuple(models),
        threshold=threshold,
    )