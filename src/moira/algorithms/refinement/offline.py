"""Offline structural refinement for concept graphs."""

from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Callable, Iterable
from itertools import chain

from moira.algorithms.graph import (
    ConceptGraph,
    EquivalentClass,
    EquivalentClassRelation,
)

ConceptSimilarity = Callable[[EquivalentClass, EquivalentClass], bool]


class OfflineGraphRefiner:
    """Compute the maximal structurally bisimilar partition of a graph.

    Semantic similarity forms the initial rank buckets. Structural signatures
    then split those buckets to a fixed point before any nodes are collapsed.
    This preserves the partition-refinement order in KROMA Algorithm 1.
    """

    def __init__(
        self, concept_similar: ConceptSimilarity | None = None
    ) -> None:
        self.concept_similar = concept_similar or (
            lambda left, right: left.id == right.id
        )

    def refine(self, graph: ConceptGraph) -> ConceptGraph:
        refined = copy.deepcopy(graph)
        ranks = refined.compute_ranks()
        if not ranks:
            return refined

        partitions = self._semantic_rank_partitions(refined, ranks)
        while True:
            block_of = {
                node_id: index
                for index, block in enumerate(partitions)
                for node_id in block
            }
            signatures = self._structural_signatures(refined, block_of)
            updated: list[set[str]] = []
            for block in partitions:
                groups: dict[tuple[object, ...], set[str]] = defaultdict(set)
                for node_id in block:
                    groups[signatures[node_id]].add(node_id)
                updated.extend(
                    groups[key] for key in sorted(groups, key=repr)
                )
            if self._normalized(updated) == self._normalized(partitions):
                partitions = updated
                break
            partitions = updated

        for block in sorted(partitions, key=lambda value: sorted(value)):
            if len(block) > 1:
                self.collapse(refined, block)

        refined.compute_ranks()
        return refined

    def _semantic_rank_partitions(
        self,
        graph: ConceptGraph,
        ranks: dict[str, int],
    ) -> list[set[str]]:
        """Partition each rank by the paper's transitive similarity relation."""
        by_rank: dict[int, list[str]] = defaultdict(list)
        for node_id, rank in ranks.items():
            by_rank[rank].append(node_id)

        partitions: list[set[str]] = []
        for rank in sorted(by_rank):
            for node_id in sorted(by_rank[rank]):
                node = graph.nodes[node_id]
                for block in partitions:
                    representative_id = min(block)
                    if (
                        ranks[representative_id] == rank
                        and self.concept_similar(
                            node, graph.nodes[representative_id]
                        )
                    ):
                        block.add(node_id)
                        break
                else:
                    partitions.append({node_id})
        return partitions

    @staticmethod
    def _structural_signatures(
        graph: ConceptGraph,
        block_of: dict[str, int],
    ) -> dict[str, tuple[object, ...]]:
        incoming: dict[str, set[tuple[str, int]]] = defaultdict(set)
        outgoing: dict[str, set[tuple[str, int]]] = defaultdict(set)
        for edge in graph.edges:
            outgoing[edge.src.id].add(
                (edge.relation, block_of[edge.tgt.id])
            )
            incoming[edge.tgt.id].add(
                (edge.relation, block_of[edge.src.id])
            )
        return {
            node_id: (
                tuple(sorted(incoming[node_id])),
                tuple(sorted(outgoing[node_id])),
            )
            for node_id in graph.nodes
        }

    @staticmethod
    def _normalized(partitions: Iterable[set[str]]) -> set[frozenset[str]]:
        return {frozenset(block) for block in partitions if block}

    @staticmethod
    def collapse(graph: ConceptGraph, node_ids: Iterable[str]) -> str:
        ids = sorted(set(node_ids))
        if not ids:
            raise ValueError("cannot collapse an empty node set")
        missing = [node_id for node_id in ids if node_id not in graph.nodes]
        if missing:
            raise KeyError(f"unknown graph nodes: {missing}")
        if len(ids) == 1:
            return ids[0]

        representative_id = ids[0]
        merged_concepts = list(
            chain.from_iterable(graph.nodes[node_id].equiv_concepts for node_id in ids)
        )
        merged_concepts.sort(
            key=lambda concept: (
                concept.name or "",
                concept.nodeid or "",
                concept.iri or "",
            )
        )
        merged = EquivalentClass(merged_concepts)
        merged.id = representative_id
        merged_ids = set(ids)

        edge_by_key: dict[tuple[str, str, str], EquivalentClassRelation] = {}
        for edge in graph.edges:
            source_id = representative_id if edge.src.id in merged_ids else edge.src.id
            target_id = representative_id if edge.tgt.id in merged_ids else edge.tgt.id
            if source_id == target_id:
                continue
            source = (
                merged if source_id == representative_id else graph.nodes[source_id]
            )
            target = (
                merged if target_id == representative_id else graph.nodes[target_id]
            )
            key = (source_id, target_id, edge.relation)
            candidate = EquivalentClassRelation(
                source,
                target,
                relation=edge.relation,
                score=edge.score,
            )
            previous = edge_by_key.get(key)
            if previous is None or candidate.score > previous.score:
                edge_by_key[key] = candidate

        for node_id in ids:
            del graph.nodes[node_id]
        graph.nodes[representative_id] = merged

        graph.edges.clear()
        graph.parents.clear()
        graph.children.clear()
        for node in graph.nodes.values():
            node.parents = []
            node.children = []
        for key in sorted(edge_by_key):
            graph.add_edge(edge_by_key[key])
        return representative_id
