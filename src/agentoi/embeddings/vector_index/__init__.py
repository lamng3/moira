"""Vector index contracts, registry, and built-in backends."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "VectorIndex": ("agentoi.embeddings.protocols", "VectorIndex"),
    "SearchResult": (f"{__name__}.base", "SearchResult"),
    "BaseVectorIndex": (f"{__name__}.base", "BaseVectorIndex"),
    "MultiQuerySearchService": (
        f"{__name__}.multi_query",
        "MultiQuerySearchService",
    ),
    "TorchExactVectorIndex": (f"{__name__}.torch_exact", "TorchExactVectorIndex"),
    "VectorIndexRegistry": (f"{__name__}.registry", "VectorIndexRegistry"),
    "register_vector_index": (f"{__name__}.registry", "register_vector_index"),
    "unregister_vector_index": (
        f"{__name__}.registry",
        "unregister_vector_index",
    ),
    "create_vector_index": (f"{__name__}.registry", "create_vector_index"),
    "available_vector_indexes": (
        f"{__name__}.registry",
        "available_vector_indexes",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
