"""Shared, dependency-free embedding types."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Hashable, TypeAlias

Vector: TypeAlias = Any
VectorId: TypeAlias = Hashable
Metadata: TypeAlias = Mapping[str, Any]
