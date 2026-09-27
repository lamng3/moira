"""Pluggable eviction for cache tiers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """One evictable record. A new policy only needs these fields."""

    key: tuple[str, ...]
    count: int = 0
    order: int = 0
    last_used: int = 0


class EvictionPolicy(Protocol):
    """Choose the next record to drop when a tier is over its cap."""

    def choose(self, entries: Sequence[CacheEntry]) -> CacheEntry | None:
        """Return the entry to remove, or none when the tier is empty."""


class LowestCount:
    """Drop the least-used record, then the oldest one."""

    def choose(self, entries: Sequence[CacheEntry]) -> CacheEntry | None:
        if not entries:
            return None
        return min(entries, key=lambda entry: (entry.count, entry.order))


class LeastRecentlyUsed:
    """Drop the record that was read or written longest ago."""

    def choose(self, entries: Sequence[CacheEntry]) -> CacheEntry | None:
        if not entries:
            return None
        return min(entries, key=lambda entry: (entry.last_used, entry.order))


def policy_for(name: str) -> EvictionPolicy:
    """Resolve a configured eviction name. Add a policy by registering it here."""
    policies: dict[str, EvictionPolicy] = {
        "lowest-count": LowestCount(),
        "lru": LeastRecentlyUsed(),
    }
    try:
        return policies[name]
    except KeyError as exc:
        known = ", ".join(sorted(policies))
        raise ValueError(f"Unknown eviction policy {name!r}. Known policies: {known}.") from exc
