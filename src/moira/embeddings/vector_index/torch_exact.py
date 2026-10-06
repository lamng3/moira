"""Exact in-memory cosine search implemented with Torch."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch

from ..types import Metadata, VectorId
from .base import BaseVectorIndex, SearchResult


class TorchExactVectorIndex(BaseVectorIndex):
    """A deterministic exact backend suited to tests and small collections."""

    def __init__(self, dimension: int | None = None):
        if dimension is not None and dimension <= 0:
            raise ValueError("dimension must be positive")
        self._configured_dimension = dimension
        self._dimension = dimension
        self._vectors: dict[VectorId, torch.Tensor] = {}
        self._metadata: dict[VectorId, Metadata] = {}

    @property
    def dimension(self) -> int | None:
        return self._dimension

    def add(
        self,
        ids: Sequence[VectorId],
        vectors: torch.Tensor,
        metadata: Sequence[Metadata | None] | None = None,
    ) -> None:
        ids = list(ids)
        matrix = torch.as_tensor(vectors)
        if matrix.ndim == 1:
            matrix = matrix.unsqueeze(0)
        if matrix.ndim != 2:
            raise ValueError("vectors must have shape [count, dimension]")
        if len(ids) != matrix.shape[0]:
            raise ValueError("ids and vectors must have the same length")
        if len(set(ids)) != len(ids):
            raise ValueError("ids must be unique within a batch")

        metadata_values = list(metadata) if metadata is not None else [None] * len(ids)
        if len(metadata_values) != len(ids):
            raise ValueError("metadata and ids must have the same length")

        dimension = int(matrix.shape[1])
        if self._dimension is None:
            self._dimension = dimension
        elif dimension != self._dimension:
            raise ValueError(
                f"Expected vectors of dimension {self._dimension}, got {dimension}"
            )

        normalized = matrix.detach().to(dtype=torch.float32, device="cpu")
        normalized = normalized / (
            normalized.norm(p=2, dim=1, keepdim=True) + 1e-9
        )
        for item_id, vector, item_metadata in zip(
            ids, normalized, metadata_values
        ):
            self._vectors[item_id] = vector.clone()
            self._metadata[item_id] = dict(item_metadata or {})

    def remove(self, ids: Iterable[VectorId]) -> None:
        for item_id in ids:
            self._vectors.pop(item_id, None)
            self._metadata.pop(item_id, None)

    def search(
        self,
        query: torch.Tensor,
        k: int = 10,
        *,
        allowed_ids: set[VectorId] | None = None,
    ) -> list[SearchResult]:
        if k < 0:
            raise ValueError("k must be non-negative")
        if k == 0 or not self._vectors:
            return []

        vector = torch.as_tensor(query)
        if vector.ndim != 1:
            raise ValueError("query must have shape [dimension]")
        if vector.shape[0] != self._dimension:
            raise ValueError(
                f"Expected query dimension {self._dimension}, got {vector.shape[0]}"
            )

        normalized_query = vector.detach().to(dtype=torch.float32, device="cpu")
        normalized_query = normalized_query / (normalized_query.norm(p=2) + 1e-9)
        ids = [
            item_id
            for item_id in self._vectors
            if allowed_ids is None or item_id in allowed_ids
        ]
        if not ids:
            return []
        matrix = torch.stack([self._vectors[item_id] for item_id in ids])
        scores = torch.mv(matrix, normalized_query).tolist()
        ranked = sorted(
            enumerate(scores),
            key=lambda pair: (-pair[1], pair[0]),
        )[:k]
        return [
            SearchResult(
                id=ids[position],
                score=float(score),
                metadata=dict(self._metadata[ids[position]]),
            )
            for position, score in ranked
        ]

    def search_many(
        self,
        queries: Sequence[torch.Tensor],
        k: int = 10,
        *,
        agg: str = "max",
        weights: Sequence[float] | None = None,
        allowed_ids: set[VectorId] | None = None,
    ) -> list[SearchResult]:
        """Exactly aggregate scores over every indexed vector."""
        if k < 0:
            raise ValueError("k must be non-negative")
        query_list = list(queries)
        if k == 0 or not query_list or not self._vectors:
            return []

        query_matrix = torch.stack(
            [torch.as_tensor(query) for query in query_list],
            dim=0,
        )
        if query_matrix.ndim != 2:
            raise ValueError("queries must have shape [count, dimension]")
        if query_matrix.shape[1] != self._dimension:
            raise ValueError(
                f"Expected query dimension {self._dimension}, "
                f"got {query_matrix.shape[1]}"
            )

        normalized_queries = query_matrix.detach().to(
            dtype=torch.float32,
            device="cpu",
        )
        normalized_queries = normalized_queries / (
            normalized_queries.norm(p=2, dim=1, keepdim=True) + 1e-9
        )
        ids = [
            item_id
            for item_id in self._vectors
            if allowed_ids is None or item_id in allowed_ids
        ]
        if not ids:
            return []
        matrix = torch.stack([self._vectors[item_id] for item_id in ids])
        scores = normalized_queries @ matrix.T

        if agg == "mean":
            aggregated = scores.mean(dim=0)
        elif agg == "sum":
            aggregated = scores.sum(dim=0)
        elif agg == "weighted":
            if weights is None or len(weights) != len(query_list):
                query_weights = torch.full(
                    (len(query_list),),
                    1.0 / len(query_list),
                    dtype=scores.dtype,
                )
            else:
                query_weights = torch.as_tensor(weights, dtype=scores.dtype)
                query_weights = query_weights / (query_weights.sum() + 1e-9)
            aggregated = (query_weights[:, None] * scores).sum(dim=0)
        else:
            aggregated = scores.max(dim=0).values

        count = min(k, len(ids))
        values, positions = torch.topk(
            aggregated,
            k=count,
            largest=True,
            sorted=True,
        )
        return [
            SearchResult(
                id=ids[int(position)],
                score=float(value),
                metadata=dict(self._metadata[ids[int(position)]]),
            )
            for value, position in zip(values.tolist(), positions.tolist())
        ]

    def clear(self) -> None:
        self._vectors.clear()
        self._metadata.clear()
        self._dimension = self._configured_dimension

    def __len__(self) -> int:
        return len(self._vectors)
