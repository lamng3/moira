"""Shared ontology-to-structure building primitives."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

from .predicates import is_equivalence_predicate


class OntologyStructureBuilder:
    """Collapse ontology equivalences once for graph and hypergraph builders."""

    @staticmethod
    def equivalence_classes(
        ontology: Any,
        *,
        dsu_factory: Callable[[], Any],
        equivalent_class_factory: Callable[[list[Any]], Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        dsu = dsu_factory()
        for edge in ontology.edges:
            if is_equivalence_predicate(edge.pred):
                dsu.union(edge.src.nodeid, edge.tgt.nodeid)
        for concept in ontology.nodes:
            dsu.find(concept.nodeid)

        groups: dict[str, list[Any]] = defaultdict(list)
        for concept in ontology.nodes:
            groups[dsu.find(concept.nodeid)].append(concept)

        nodes: dict[str, Any] = {}
        concept_to_equivalence: dict[str, Any] = {}
        for root, members in groups.items():
            members.sort(key=lambda concept: (concept.name or "", concept.nodeid))
            equivalent_class = equivalent_class_factory(members)
            equivalent_class.id = f"eq_{root}"
            nodes[equivalent_class.id] = equivalent_class
            for member in members:
                concept_to_equivalence[member.nodeid] = equivalent_class
        return nodes, concept_to_equivalence
