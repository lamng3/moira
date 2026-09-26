"""Embedding encoder implementations, loaded only when requested."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "TextEmbedding": "text",
    "GraphEmbedding": "graph",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value
