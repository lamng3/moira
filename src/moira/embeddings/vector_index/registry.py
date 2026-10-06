"""Registry and factory for vector-index backends."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import BaseVectorIndex

VectorIndexFactory = Callable[..., BaseVectorIndex]


class VectorIndexRegistry:
    def __init__(self):
        self._factories: dict[str, VectorIndexFactory] = {}

    def register(
        self,
        name: str,
        factory: VectorIndexFactory,
        *,
        replace: bool = False,
    ) -> None:
        key = self._key(name)
        if key in self._factories and not replace:
            raise ValueError(f"Vector index backend already registered: {name!r}")
        self._factories[key] = factory

    def unregister(self, name: str) -> None:
        self._factories.pop(self._key(name), None)

    def create(self, name: str, **kwargs: Any) -> BaseVectorIndex:
        key = self._key(name)
        try:
            factory = self._factories[key]
        except KeyError as error:
            available = ", ".join(self.available()) or "none"
            raise ValueError(
                f"Unknown vector index backend {name!r}; available: {available}"
            ) from error
        return factory(**kwargs)

    def available(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    @staticmethod
    def _key(name: str) -> str:
        key = name.strip().lower()
        if not key:
            raise ValueError("Backend name cannot be empty")
        return key


def _create_torch_exact(**kwargs: Any) -> BaseVectorIndex:
    from .torch_exact import TorchExactVectorIndex

    return TorchExactVectorIndex(**kwargs)


registry = VectorIndexRegistry()
registry.register("torch-exact", _create_torch_exact)


def register_vector_index(
    name: str,
    factory: VectorIndexFactory,
    *,
    replace: bool = False,
) -> None:
    registry.register(name, factory, replace=replace)


def unregister_vector_index(name: str) -> None:
    registry.unregister(name)


def create_vector_index(name: str = "torch-exact", **kwargs: Any) -> BaseVectorIndex:
    return registry.create(name, **kwargs)


def available_vector_indexes() -> tuple[str, ...]:
    return registry.available()
