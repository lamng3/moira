"""Ontology translation and tuple-validation orchestration."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from moira.algorithms.graph import ConceptGraph, Ontology

from .tuples import (
    EvaluationStatistics,
    OntologyTuple,
    TupleEvaluation,
    TupleEvaluator,
    TupleNormalizer,
)


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Immutable output from tuple normalization and evaluation."""

    original: tuple[OntologyTuple, ...]
    normalized: tuple[OntologyTuple, ...]
    evaluations: tuple[TupleEvaluation, ...]
    statistics: EvaluationStatistics


@dataclass(frozen=True, slots=True)
class OntologyPipelineResult:
    """Tuple results together with translated graph models."""

    tuples: PipelineResult
    ontology: Ontology
    concept_graph: ConceptGraph


class OntologyTranslator:
    """Translate triples through MOIRA's real graph models."""

    def __init__(self) -> None:
        self._ontology_cache: dict[
            tuple[tuple[OntologyTuple, ...], str, str], Ontology
        ] = {}
        self._graph_cache: dict[
            tuple[str, str, tuple[OntologyTuple, ...]], ConceptGraph
        ] = {}

    def triples_to_ontology(
        self,
        triples: Iterable[OntologyTuple],
        *,
        name: str = "ProcessedOntology",
        version: str = "1.0.0",
    ) -> Ontology:
        materialized = tuple(triples)
        key = materialized, name, version
        if key not in self._ontology_cache:
            ontology = Ontology(name=name, version=version)
            ontology.build_ontology_from_triples(list(materialized))
            self._ontology_cache[key] = ontology
        return self._ontology_cache[key]

    def ontology_to_concept_graph(self, ontology: Ontology) -> ConceptGraph:
        triples = tuple(
            (edge.src.name, edge.pred, edge.tgt.name) for edge in ontology.edges
        )
        key = ontology.name, ontology.version, triples
        if key not in self._graph_cache:
            graph = ConceptGraph(nodes=[], edges=[])
            graph.build_from_ontology(ontology)
            self._graph_cache[key] = graph
        return self._graph_cache[key]


class ValidationPipeline:
    """Normalize, evaluate, and optionally translate ontology tuples."""

    def __init__(
        self,
        *,
        normalizer: TupleNormalizer | None = None,
        evaluator: TupleEvaluator | None = None,
        translator: OntologyTranslator | None = None,
    ) -> None:
        self.normalizer = normalizer or TupleNormalizer()
        self.evaluator = evaluator or TupleEvaluator()
        self.translator = translator or OntologyTranslator()

    def process(self, values: Iterable[OntologyTuple]) -> PipelineResult:
        original = tuple(values)
        normalized = self.normalizer.normalize_all(original)
        evaluations = self.evaluator.evaluate_all(normalized)
        return PipelineResult(
            original=original,
            normalized=normalized,
            evaluations=evaluations,
            statistics=self.evaluator.statistics(evaluations),
        )

    def process_ontology(
        self,
        triples: Iterable[OntologyTuple],
        *,
        name: str = "ProcessedOntology",
        version: str = "1.0.0",
    ) -> OntologyPipelineResult:
        materialized = tuple(triples)
        ontology = self.translator.triples_to_ontology(
            materialized, name=name, version=version
        )
        return OntologyPipelineResult(
            tuples=self.process(materialized),
            ontology=ontology,
            concept_graph=self.translator.ontology_to_concept_graph(ontology),
        )
