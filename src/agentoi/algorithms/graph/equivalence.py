from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .concept import Concept

if TYPE_CHECKING:
    import torch


class EquivalentClass:
    """bisimilar concepts treated as one equivalence class"""

    def __init__(
        self,
        equiv_concepts: list[Concept],
        parents: list[EquivalentClass] | None = None,
        children: list[EquivalentClass] | None = None,
    ):
        if not equiv_concepts:
            raise ValueError("EquivalentClass requires at least one Concept")
        self.equiv_concepts = equiv_concepts
        self.id: str = equiv_concepts[0].name  # use first concept name as id
        self.rank: int | None = None
        self.parents: list[EquivalentClass] = parents if parents is not None else []
        self.children: list[EquivalentClass] = children if children is not None else []
        self.text_embedding: torch.Tensor = None
        self.graph_embedding: torch.Tensor = None
        self.embedding: torch.Tensor = None

    def compute_embedding(self, alpha: float = 0.5) -> torch.Tensor:
        """
        fuse graph- and text-embeddings into a single vector
            z_c = alpha * graph_emb + (1-alpha) * text_emb
        cache results in self.embedding
        """
        if hasattr(self, "embedding") and self.embedding is not None:
            return self.embedding
        from agentoi.embeddings import TextEmbedding, fuse_embeddings

        if self.graph_embedding is None:
            raise RuntimeError(
                "Graph embedding is unavailable. Call the containing graph's "
                "compute_all_embeddings() before requesting a fused embedding."
            )
        # ensure text_embedding exists
        if self.text_embedding is None:
            # compute & cache
            self.text_embedding = TextEmbedding().compute_embedding(self)
            if self.text_embedding is None:
                raise RuntimeError("TextEmbedding returned None")
        z = fuse_embeddings(self.graph_embedding, self.text_embedding, alpha)
        self.embedding = z
        return z

    def compute_rank(self) -> int:
        """Return the cached local rank.

        Graph mutations should be followed by ``ConceptGraph.compute_ranks()``,
        which refreshes every class consistently.
        """
        if self.rank is None:
            self.rank = 1 + max(
                (child.compute_rank() for child in self.children), default=-1
            )
        return self.rank

    def members(self, return_label=False) -> str:
        """get members within an equivalent class"""
        if return_label:
            concept_labels: list[str] = []
            for c in self.equiv_concepts:
                if label := (
                    c.ground_set.get("labels") or c.ground_set.get("alt_labels")
                ):
                    concept_labels.append(label[0])
                else:
                    concept_labels.append(c.name.split("/")[-1])
            return ", ".join(concept_labels)

        return ", ".join(c.name for c in self.equiv_concepts)

    def describe(self) -> str:
        """narrative summary of an equivalent class and its relations"""
        members = self.members() or "none"
        parents = ", ".join(p.members() for p in self.parents) or "none"
        children = ", ".join(c.members() for c in self.children) or "none"
        return (
            f"This equivalence class, comprised of {members}, "
            "brings together semantically aligned concepts into a unified whole. "
            f"It is rooted in the broader notions of {parents}, from which it descends, "
            f"and it branches out to give rise to {children} as its sub-concepts."
        )

    def __repr__(self) -> str:
        names = [c.name for c in self.equiv_concepts]
        labels = [c.ground_set["labels"] for c in self.equiv_concepts]
        return f"EquivalentClass(id={self.id!r}, concepts={names} -- {labels})"


class EquivalentClassRelation:
    """relations between equivalent classes"""

    def __init__(
        self,
        src: EquivalentClass,
        tgt: EquivalentClass,
        relation: str,
        score: float = 0.0,
    ):
        assert relation in ("yes", "no"), "relation must be 'yes' or 'no'"
        self.src = src
        self.tgt = tgt
        self.relation = relation
        self.score = score

    def describe(self) -> str:
        source_summary = self.src.describe()
        target_summary = self.tgt.describe()
        relation_label = (
            "semantically equivalent"
            if self.relation == "yes"
            else "not semantically equivalent"
        )
        return (
            f"Equivalence class A: {source_summary}; "
            f"Equivalence class B: {target_summary}. "
            f"Equivalence class A are {relation_label} to equivalence class B."
        )

    def __repr__(self) -> str:
        return (
            f"EquivalentClassRelation("
            f"src={self.src!r}, "
            f"tgt={self.tgt!r}, "
            f"score={self.score})"
        )


class DSU:
    """disjoint set union with path compression and union by size to check cycle inside ontology"""

    def __init__(self):
        self.parent: dict[Any, Any] = {}
        self.size: dict[Any, int] = {}

    def find(self, v):
        if v not in self.parent:  # lazy create v
            self.parent[v] = v
            self.size[v] = 1
            return v
        if self.parent[v] != v:
            # path compression -- inverse ackermann time complexity
            self.parent[v] = self.find(self.parent[v])
        return self.parent[v]

    def union(self, a, b):
        a = self.find(a)
        b = self.find(b)
        if a == b:
            return False
        if self.size[a] < self.size[b]:
            a, b = b, a  # swap
        self.parent[b] = a
        self.size[a] += self.size[b]
        return True
