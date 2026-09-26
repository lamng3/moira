"""Refinement adapter for ontology objects."""

from __future__ import annotations

from dataclasses import replace

from agentoi.algorithms.graph import Ontology

from .pipeline import (
    RefinementDiagnostic,
    RefinementOptions,
    RefinementResult,
    RelationRefiner,
)


class OntologyRefiner:
    """Apply relation refinement and rebuild an ontology with its metadata."""

    def __init__(self, relation_refiner: RelationRefiner | None = None) -> None:
        self.relation_refiner = relation_refiner or RelationRefiner()
        self.last_result: RefinementResult | None = None

    def refine(
        self,
        ontology: Ontology,
        options: RefinementOptions | None = None,
        *,
        fallback_if_expanded: bool = False,
    ) -> Ontology:
        refined, result = self.refine_with_result(
            ontology,
            options,
            fallback_if_expanded=fallback_if_expanded,
        )
        self.last_result = result
        return refined

    def refine_with_result(
        self,
        ontology: Ontology,
        options: RefinementOptions | None = None,
        *,
        fallback_if_expanded: bool = False,
    ) -> tuple[Ontology, RefinementResult]:
        opts = options or self.relation_refiner.options
        triples = [(edge.src.name, edge.pred, edge.tgt.name) for edge in ontology.edges]
        result = self.relation_refiner.refine(triples, opts)

        if fallback_if_expanded and len(result.triples) > len(ontology.edges):
            result = self.relation_refiner.refine(
                triples, replace(opts, transitive_closure=False)
            )
            result.diagnostics.insert(
                0,
                RefinementDiagnostic(
                    "ontology",
                    "closure_fallback",
                    "Refinement was repeated without closure to avoid graph expansion.",
                    details={
                        "before": len(ontology.edges),
                        "after": len(result.triples),
                    },
                ),
            )
            result.actions.insert(0, "Repeated refinement without transitive closure.")
            result.metadata["closure_fallback"] = True

        refined = Ontology(
            name=ontology.name,
            version=ontology.version,
        ).build_ontology_from_triples(result.triples)
        self._copy_metadata(ontology, refined)
        self.last_result = result
        return refined, result

    @staticmethod
    def _copy_metadata(source: Ontology, target: Ontology) -> None:
        source_by_name = {node.name: node for node in source.nodes}
        for node in target.nodes:
            original = source_by_name.get(node.name)
            if original is None:
                continue
            node.iri = original.iri
            node.ground_set = Ontology.copy_ground_set(original.ground_set)
