"""Core vector-index result type and abstract contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from math import fsum

from ..types import Metadata, Vector, VectorId


@dataclass(frozen=True, slots=True)
class SearchResult:
    id: VectorId
    score: float
    metadata: Metadata = field(default_factory=dict)


class BaseVectorIndex(ABC):
    """Minimal lifecycle shared by local and future remote index adapters."""

    @property
    @abstractmethod
    def dimension(self) -> int | None:
        raise NotImplementedError

    @abstractmethod
    def add(
        self,
        ids: Sequence[VectorId],
        vectors: Vector,
        metadata: Sequence[Metadata | None] | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def remove(self, ids: Iterable[VectorId]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query: Vector,
        k: int = 10,
        *,
        allowed_ids: set[VectorId] | None = None,
    ) -> list[SearchResult]:
        raise NotImplementedError

    def search_many(
        self,
        queries: Sequence[Vector],
        k: int = 10,
        *,
        agg: str = "max",
        weights: Sequence[float] | None = None,
        allowed_ids: set[VectorId] | None = None,
    ) -> list[SearchResult]:
        """Aggregate multiple queries after retrieving every allowed candidate.

        ANN and remote backends may override this method with a native
        multi-query implementation. The fallback deliberately requests the
        complete allowed set for every query so aggregation never drops a
        candidate.
        """
        query_list = list(queries)
        if k < 0:
            raise ValueError("k must be non-negative")
        if not query_list or k == 0 or not self:
            return []

        candidate_count = len(allowed_ids) if allowed_ids is not None else len(self)
        result_sets = [
            self.search(query, k=candidate_count, allowed_ids=allowed_ids)
            for query in query_list
        ]
        by_id = [
            {result.id: result for result in results} for results in result_sets
        ]
        ordered_ids = [result.id for result in result_sets[0]]
        if any(len(results) != candidate_count for results in result_sets):
            raise RuntimeError(
                "search_many requires every query to return all indexed candidates"
            )

        if agg == "weighted":
            if weights is None or len(weights) != len(query_list):
                normalized_weights = [1.0 / len(query_list)] * len(query_list)
            else:
                total = fsum(weights)
                denominator = total + 1e-9
                normalized_weights = [weight / denominator for weight in weights]

        aggregated: list[SearchResult] = []
        for item_id in ordered_ids:
            scores = [results[item_id].score for results in by_id]
            if agg == "mean":
                score = fsum(scores) / len(scores)
            elif agg == "sum":
                score = fsum(scores)
            elif agg == "weighted":
                score = fsum(
                    weight * value
                    for weight, value in zip(normalized_weights, scores)
                )
            else:
                score = max(scores)
            aggregated.append(
                SearchResult(
                    id=item_id,
                    score=score,
                    metadata=by_id[0][item_id].metadata,
                )
            )

        return sorted(aggregated, key=lambda result: -result.score)[:k]

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError
