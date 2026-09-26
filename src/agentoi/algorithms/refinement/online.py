"""Incremental ontology refinement for streamed graph updates."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum

from agentoi.algorithms.graph import (
    ConceptGraph,
    EquivalentClass,
    EquivalentClassRelation,
)

from .offline import OfflineGraphRefiner

ConceptSimilarity = Callable[[EquivalentClass, EquivalentClass], bool]


class DeferredReason(str, Enum):
    """Reasons an online update needs external validation."""

    LOW_CONFIDENCE = "low_confidence"
    SEMANTIC_CONFLICT = "semantic_conflict"
    INCONSISTENT_RANK = "inconsistent_rank"
    CYCLE = "cycle"


@dataclass(frozen=True)
class GraphUpdate:
    """One directed relation arriving from an ontology stream."""

    source: EquivalentClass
    target: EquivalentClass
    relation: str = "no"
    score: float = 2.0
    confidence: float | None = None
    semantically_similar: bool | None = None


@dataclass(frozen=True)
class DeferredUpdate:
    update: GraphUpdate
    reason: DeferredReason
    message: str


@dataclass
class OnlineRefinementResult:
    graph: ConceptGraph
    applied: list[GraphUpdate] = field(default_factory=list)
    deferred: list[DeferredUpdate] = field(default_factory=list)
    merged_classes: list[tuple[str, ...]] = field(default_factory=list)
    split_classes: list[tuple[str, str]] = field(default_factory=list)


class OnlineOntologyRefiner:
    """Maintain a concept DAG as triples arrive in small batches.

    The implementation follows KROMA Algorithm 2: updates are processed from
    lower to higher rank, compatible local classes are merged, and uncertain or
    rank-inconsistent updates are returned for expert validation.
    """

    def __init__(
        self,
        *,
        concept_similar: ConceptSimilarity | None = None,
        confidence_threshold: float = 8.5,
    ) -> None:
        self.concept_similar = concept_similar or (
            lambda left, right: left.id == right.id
        )
        self.confidence_threshold = float(confidence_threshold)

    def refine(
        self,
        graph: ConceptGraph,
        updates: Iterable[GraphUpdate],
        *,
        copy_graph: bool = True,
    ) -> OnlineRefinementResult:
        refined = copy.deepcopy(graph) if copy_graph else graph
        result = OnlineRefinementResult(graph=refined)
        update_list = list(updates)
        projected = copy.deepcopy(refined)
        projected_source_ids: dict[int, str] = {}
        for position, update in enumerate(update_list):
            source, _, _ = self._resolve_endpoint(projected, update.source)
            target, _, _ = self._resolve_endpoint(projected, update.target)
            projected_source_ids[position] = source.id
            if (
                source.id != target.id
                and not self._edge_exists(
                    projected, source.id, target.id, update.relation
                )
            ):
                projected.add_edge(
                    EquivalentClassRelation(
                        source,
                        target,
                        relation=update.relation,
                        score=update.score,
                    )
                )
        projected_ranks = projected.compute_ranks()
        ordered = [
            update
            for position, update in sorted(
                enumerate(update_list),
                key=lambda item: (
                    projected_ranks.get(projected_source_ids[item[0]], 0),
                    item[1].source.id,
                    item[1].target.id,
                ),
            )
        ]

        for update in ordered:
            if (
                update.confidence is not None
                and update.confidence < self.confidence_threshold
            ):
                result.deferred.append(
                    DeferredUpdate(
                        update,
                        DeferredReason.LOW_CONFIDENCE,
                        "Semantic confidence is below the acceptance threshold.",
                    )
                )
                continue

            source, _source_keys, source_known = self._resolve_endpoint(
                refined, update.source
            )
            target, target_keys, target_known = self._resolve_endpoint(
                refined, update.target
            )
            semantically_similar = (
                update.semantically_similar
                if update.semantically_similar is not None
                else self.concept_similar(source, target)
            )

            if source.id == target.id and not semantically_similar:
                extracted = self._split_endpoint(
                    refined,
                    target.id,
                    target_keys,
                    preferred_id=update.target.id,
                )
                if extracted.id != source.id:
                    result.split_classes.append((source.id, extracted.id))
                    target = extracted
                    target_known = False
                result.deferred.append(
                    DeferredUpdate(
                        update,
                        DeferredReason.SEMANTIC_CONFLICT,
                        "A non-similar endpoint was split from its equivalence class.",
                    )
                )
                continue

            ranks = refined.compute_ranks()
            source_rank = ranks.get(source.id, 0)
            target_rank = ranks.get(target.id, 0)
            target_parents = tuple(refined.parents_of(target))
            source_children = tuple(refined.children_of(source))

            if self._path_exists(refined, target.id, source.id):
                result.deferred.append(
                    DeferredUpdate(
                        update,
                        DeferredReason.CYCLE,
                        "Applying the update would create a cycle.",
                    )
                )
                continue
            if source_known and target_known and source_rank < target_rank:
                result.deferred.append(
                    DeferredUpdate(
                        update,
                        DeferredReason.INCONSISTENT_RANK,
                        "The parent endpoint ranks below the child endpoint.",
                    )
                )
                continue

            if not self._edge_exists(refined, source.id, target.id, update.relation):
                refined.add_edge(
                    EquivalentClassRelation(
                        source,
                        target,
                        relation=update.relation,
                        score=update.score,
                    )
                )
            result.applied.append(update)
            refined.compute_ranks()
            if source_rank > target_rank:
                result.merged_classes.extend(
                    self._merge_rank_bucket(
                        refined, source.id, source_rank
                    )
                )
                result.merged_classes.extend(
                    self._merge_rank_bucket(
                        refined, target.id, target_rank
                    )
                )
            elif source_rank == target_rank:
                result.merged_classes.extend(
                    self._merge_candidates(
                        refined,
                        source.id,
                        (node.id for node in target_parents),
                    )
                )
                result.merged_classes.extend(
                    self._merge_candidates(
                        refined,
                        target.id,
                        (node.id for node in source_children),
                    )
                )

        refined.compute_ranks()
        return result

    @staticmethod
    def _resolve_endpoint(
        graph: ConceptGraph, incoming: EquivalentClass
    ) -> tuple[EquivalentClass, set[str], bool]:
        keys = OnlineOntologyRefiner._concept_keys(incoming)
        existing = graph.nodes.get(incoming.id)
        if existing is not None:
            return existing, keys, True
        for node in graph.nodes.values():
            if keys & OnlineOntologyRefiner._concept_keys(node):
                return node, keys, True
        graph.add_node(copy.deepcopy(incoming))
        return graph.nodes[incoming.id], keys, False

    @staticmethod
    def _edge_exists(
        graph: ConceptGraph,
        source_id: str,
        target_id: str,
        relation: str,
    ) -> bool:
        return any(
            edge.src.id == source_id
            and edge.tgt.id == target_id
            and edge.relation == relation
            for edge in graph.edges
        )

    @staticmethod
    def _path_exists(graph: ConceptGraph, source_id: str, target_id: str) -> bool:
        pending = [source_id]
        visited: set[str] = set()
        while pending:
            current = pending.pop()
            if current == target_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            pending.extend(node.id for node in graph.children.get(current, ()))
        return False

    def _merge_rank_bucket(
        self,
        graph: ConceptGraph,
        node_id: str,
        rank: int,
    ) -> list[tuple[str, ...]]:
        ranks = graph.compute_ranks()
        return self._merge_candidates(
            graph,
            node_id,
            (
                candidate_id
                for candidate_id, candidate_rank in ranks.items()
                if candidate_rank == rank
            ),
        )

    def _merge_candidates(
        self,
        graph: ConceptGraph,
        node_id: str,
        candidate_ids: Iterable[str],
    ) -> list[tuple[str, ...]]:
        node = graph.nodes.get(node_id)
        if node is None:
            return []
        candidates = [
            graph.nodes[candidate_id]
            for candidate_id in sorted(set(candidate_ids))
            if candidate_id in graph.nodes and candidate_id != node.id
        ]
        compatible = [
            candidate
            for candidate in candidates
            if self.concept_similar(node, candidate)
            and self._structurally_equivalent(graph, node, candidate)
        ]
        if not compatible:
            return []
        ids = tuple(sorted({node.id, *(item.id for item in compatible)}))
        OfflineGraphRefiner.collapse(graph, ids)
        return [ids]

    @staticmethod
    def _concept_keys(node: EquivalentClass) -> set[str]:
        keys: set[str] = set()
        for concept in node.equiv_concepts:
            for value in (concept.nodeid, concept.iri, concept.name):
                if value:
                    keys.add(value)
        return keys

    @staticmethod
    def _split_endpoint(
        graph: ConceptGraph,
        class_id: str,
        endpoint_keys: set[str],
        *,
        preferred_id: str,
    ) -> EquivalentClass:
        original = graph.nodes[class_id]
        selected = [
            concept
            for concept in original.equiv_concepts
            if endpoint_keys
            & {
                value
                for value in (concept.nodeid, concept.iri, concept.name)
                if value
            }
        ]
        remaining = [
            concept
            for concept in original.equiv_concepts
            if concept not in selected
        ]
        if not selected or not remaining:
            return original

        retained = EquivalentClass(remaining)
        retained.id = class_id
        extracted = EquivalentClass(selected)
        extracted_id = preferred_id
        if not extracted_id or extracted_id == class_id or extracted_id in graph.nodes:
            extracted_id = selected[0].nodeid or selected[0].name
        if extracted_id == class_id or extracted_id in graph.nodes:
            extracted_id = f"{class_id}::split"
        extracted.id = extracted_id

        graph.nodes[class_id] = retained
        graph.nodes[extracted_id] = extracted
        rebuilt: dict[
            tuple[str, str, str], EquivalentClassRelation
        ] = {}
        for edge in graph.edges:
            sources = (
                (retained, extracted) if edge.src.id == class_id else (edge.src,)
            )
            targets = (
                (retained, extracted) if edge.tgt.id == class_id else (edge.tgt,)
            )
            for source in sources:
                for target in targets:
                    if source.id == target.id:
                        continue
                    key = (source.id, target.id, edge.relation)
                    candidate = EquivalentClassRelation(
                        source,
                        target,
                        relation=edge.relation,
                        score=edge.score,
                    )
                    previous = rebuilt.get(key)
                    if previous is None or candidate.score > previous.score:
                        rebuilt[key] = candidate

        graph.edges.clear()
        graph.parents.clear()
        graph.children.clear()
        for node in graph.nodes.values():
            node.parents = []
            node.children = []
        for key in sorted(rebuilt):
            graph.add_edge(rebuilt[key])
        return extracted

    @staticmethod
    def _structurally_equivalent(
        graph: ConceptGraph,
        left: EquivalentClass,
        right: EquivalentClass,
    ) -> bool:
        pair = {left.id, right.id}
        left_parents = {node.id for node in graph.parents_of(left)} - pair
        right_parents = {node.id for node in graph.parents_of(right)} - pair
        left_children = {node.id for node in graph.children_of(left)} - pair
        right_children = {node.id for node in graph.children_of(right)} - pair
        return left_parents == right_parents and left_children == right_children
