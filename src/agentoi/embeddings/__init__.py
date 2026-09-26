"""Public embedding API.

Implementations are imported on first attribute access so lightweight graph
imports do not initialize Torch, Transformers, or Node2Vec.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "TextEmbedding": ("encoders", "TextEmbedding"),
    "GraphEmbedding": ("encoders", "GraphEmbedding"),
    "TextEncoder": ("protocols", "TextEncoder"),
    "GraphEncoder": ("protocols", "GraphEncoder"),
    "VectorIndex": ("protocols", "VectorIndex"),
    "Vector": ("types", "Vector"),
    "VectorId": ("types", "VectorId"),
    "Metadata": ("types", "Metadata"),
    "weighted_fusion": ("fusion", "weighted_fusion"),
    "fuse_embeddings": ("fusion", "fuse_embeddings"),
    "SearchResult": ("vector_index", "SearchResult"),
    "BaseVectorIndex": ("vector_index", "BaseVectorIndex"),
    "MultiQuerySearchService": ("vector_index", "MultiQuerySearchService"),
    "TorchExactVectorIndex": ("vector_index", "TorchExactVectorIndex"),
    "VectorIndexRegistry": ("vector_index", "VectorIndexRegistry"),
    "register_vector_index": ("vector_index", "register_vector_index"),
    "unregister_vector_index": ("vector_index", "unregister_vector_index"),
    "create_vector_index": ("vector_index", "create_vector_index"),
    "available_vector_indexes": ("vector_index", "available_vector_indexes"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value
