"""Shared public types for similarity scoring."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentoi.algorithms.graph import EquivalentClass


class SimilarityMethod(Enum):
    """Available node-pair similarity methods."""

    COSINE = "cosine"
    GOOGLE_INDEX = "google_index"
    HYBRID = "hybrid"


@dataclass
class SimilarityResult:
    """A similarity score for a pair of nodes."""

    method: SimilarityMethod
    node_id_1: str
    node_id_2: str
    similarity_score: float
    metadata: dict[str, Any] | None = None


@dataclass
class ComparisonResult:
    """All supported similarity scores and ranks for a node pair."""

    node_id_1: str
    node_id_2: str
    cosine_similarity: float
    google_similarity: float
    hybrid_similarity: float
    cosine_rank: int
    google_rank: int
    hybrid_rank: int
    metadata: dict[str, Any] | None = None


@dataclass
class CrossGraphPair:
    """A scored pair containing one node from each graph."""

    node_a_id: str
    node_b_id: str
    node_a: EquivalentClass
    node_b: EquivalentClass
    similarity_score: float
    method: str
    metadata: dict[str, Any] | None = None
