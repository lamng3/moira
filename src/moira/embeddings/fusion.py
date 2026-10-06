"""Utilities for combining embedding vectors."""

from __future__ import annotations

from collections.abc import Sequence

from .types import Vector


def weighted_fusion(vectors: Sequence[Vector], weights: Sequence[float]) -> Vector:
    """Return a weighted sum of equally shaped tensors."""
    if not vectors:
        raise ValueError("At least one embedding is required")
    if len(vectors) != len(weights):
        raise ValueError("vectors and weights must have the same length")
    if any(vector.shape != vectors[0].shape for vector in vectors[1:]):
        shapes = [tuple(vector.shape) for vector in vectors]
        raise ValueError(f"Embedding dimensions must match: {shapes}")

    result = vectors[0] * weights[0]
    for vector, weight in zip(vectors[1:], weights[1:]):
        result = result + vector * weight
    return result


def fuse_embeddings(
    graph_embedding: Vector,
    text_embedding: Vector,
    alpha: float = 0.5,
) -> Vector:
    """Fuse graph and text vectors using the project's established weighting."""
    return weighted_fusion(
        (graph_embedding, text_embedding),
        (alpha, 1.0 - alpha),
    )
