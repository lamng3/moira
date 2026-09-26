from __future__ import annotations

import uuid
from collections import defaultdict

from .identifiers import name_from_uuid
from .predicates import canonicalize_predicate, is_hierarchy_predicate


class Concept:
    """concepts of an ontology"""

    def __init__(
        self,
        name: str,
        nodeid: str | None = None,
        parents: list[Concept] | None = None,
        children: list[Concept] | None = None,
        ground_set: dict[str, list[str]] | None = None,
        iri: str | None = None,
    ):
        self.name = name
        self.nodeid: str = (
            nodeid if nodeid is not None else name_from_uuid(uuid.uuid4(), "slug")
        )
        self.iri: str | None = iri
        self.rank: int | None = None
        self.parents: list[Concept] = parents if parents is not None else []
        self.children: list[Concept] = children if children is not None else []
        self.ground_set: dict[str, list[str]] = (
            ground_set
            if ground_set is not None
            else {
                "labels": [],
                "alt_labels": [],
                "related_synonyms": [],
                "exact_synonyms": [],
                "definitions": [],
            }
        )
        # multi-relational adjacency
        self.out_rels: defaultdict[str, list[Concept]] = defaultdict(list)
        self.in_rels: defaultdict[str, list[Concept]] = defaultdict(list)

    def compute_rank(self) -> int:
        """Return the cached local rank.

        Graph mutations should be followed by ``ConceptGraph.compute_ranks()``,
        which refreshes every node consistently.
        """
        if self.rank is None:
            self.rank = 1 + max(
                (child.compute_rank() for child in self.children), default=-1
            )
        return self.rank

    def add_parent(self, p: Concept):
        self.parents.append(p)

    def add_child(self, c: Concept):
        self.children.append(c)

    def _canon_pred(self, p: str) -> str:
        """normalize a predicate to lowercase local name"""
        return canonicalize_predicate(p)

    def is_is_a_predicate(self, pred: str) -> bool:
        """return true if pred is any of the 'is-a' style predicates."""
        return is_hierarchy_predicate(pred)

    def add_relation(self, pred: str, other: Concept, direction: str = "out"):
        pred = (pred or "").strip()
        if not pred:
            return

        if direction == "out":
            if other not in self.out_rels[pred]:
                self.out_rels[pred].append(other)
        else:
            if other not in self.in_rels[pred]:
                self.in_rels[pred].append(other)

        # maintain hierarchy lists in sync for "is-a" predicates
        if self.is_is_a_predicate(pred):
            if direction == "out":
                # s 'is-a' o -> s is child, o is parent
                if other not in self.parents:
                    self.parents.append(other)
                if self not in other.children:
                    other.children.append(self)
            else:
                # other 'is-a' self -> other is child, self is parent
                if other not in self.children:
                    self.children.append(other)
                if self not in other.parents:
                    other.parents.append(self)

    def __repr__(self) -> str:
        parent_names = [p.name for p in self.parents]
        child_names = [c.name for c in self.children]
        return (
            f"Concept(nodeid={self.nodeid}, name={self.name!r}, iri={self.iri!r}, "
            f"rank={self.rank}, parents={parent_names}, children={child_names}, "
            f"ground_set={self.ground_set})"
        )


class ConceptRelation:
    """relations between concepts"""

    def __init__(self, src: Concept, pred: str, tgt: Concept, score: float = 0.0):
        self.src = src
        self.pred = pred
        self.tgt = tgt
        self.score = score

    def __repr__(self) -> str:
        return f"ConceptRelation({self.src.name} -- {self.pred} --> {self.tgt.name}, score={self.score})"
