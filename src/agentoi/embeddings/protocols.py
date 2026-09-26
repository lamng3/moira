"""Structural contracts for embedding encoders and vector indexes."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .types import Metadata, Vector, VectorId

if TYPE_CHECKING:
    from .vector_index.base import SearchResult


@runtime_checkable
class TextEncoder(Protocol):
    embed_dim: int

    def to_embedding(self, text: str, max_length: int = 512) -> Vector: ...

    def compute_embedding(self, node: Any) -> Vector: ...


@runtime_checkable
class GraphEncoder(Protocol):
    dimensions: int

    def compute_embedding(self, node: Any) -> Vector: ...


@runtime_checkable
class VectorIndex(Protocol):
    """Backend-neutral lifecycle for a mutable vector index."""

    @property
    def dimension(self) -> int | None: ...

    def add(
        self,
        ids: Sequence[VectorId],
        vectors: Vector,
        metadata: Sequence[Metadata | None] | None = None,
    ) -> None: ...

    def remove(self, ids: Iterable[VectorId]) -> None: ...

    def search(
        self,
        query: Vector,
        k: int = 10,
        *,
        allowed_ids: set[VectorId] | None = None,
    ) -> list[SearchResult]: ...

    def search_many(
        self,
        queries: Sequence[Vector],
        k: int = 10,
        *,
        agg: str = "max",
        weights: Sequence[float] | None = None,
        allowed_ids: set[VectorId] | None = None,
    ) -> list[SearchResult]: ...

    def clear(self) -> None: ...

    def __len__(self) -> int: ...
