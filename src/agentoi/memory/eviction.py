"""Pluggable eviction for cache tiers."""

from __future__ import annotations

from collections import deque
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
    frequency: int = 0
    queue: str = ""


class EvictionPolicy(Protocol):
    """Choose the next record to drop when a tier is over its cap."""

    def choose(self, entries: Sequence[CacheEntry]) -> CacheEntry | None:
        """Return the entry to remove, or none when the tier is empty."""

    def label_for_insert(self, key: tuple[str, ...], *, seen: bool) -> str:
        """Queue label for a write. Empty when the policy does not use queues."""

    def label_for_hit(self, key: tuple[str, ...], label: str) -> str:
        """Queue label after a read."""

    def note_evict(self, entry: CacheEntry) -> None:
        """Remember a dropped record when the policy keeps a ghost list."""


class Eviction:
    """Shared no-op hooks. A policy overrides choose, and 2Q overrides the hooks."""

    def choose(self, entries: Sequence[CacheEntry]) -> CacheEntry | None:
        if not entries:
            return None
        return entries[0]

    def label_for_insert(self, key: tuple[str, ...], *, seen: bool) -> str:
        return ""

    def label_for_hit(self, key: tuple[str, ...], label: str) -> str:
        return label

    def note_evict(self, entry: CacheEntry) -> None:
        return None


class LeastRecentlyUsed(Eviction):
    """Drop the record that was read or written longest ago."""

    def choose(self, entries: Sequence[CacheEntry]) -> CacheEntry | None:
        if not entries:
            return None
        return min(entries, key=lambda entry: (entry.last_used, entry.order))


class LeastFrequentlyUsed(Eviction):
    """Drop the least frequently read record, then the one touched longest ago."""

    def choose(self, entries: Sequence[CacheEntry]) -> CacheEntry | None:
        if not entries:
            return None
        return min(entries, key=lambda entry: (entry.frequency, entry.last_used, entry.order))


class TwoQueue(Eviction):
    """Probation queue for one-time records, and an LRU queue for repeated ones.

    A key evicted from probation is remembered in this process. If it returns,
    it is admitted to the frequent queue. The ghost list is not saved.
    """

    def __init__(self, ghost_limit: int = 1024) -> None:
        self.ghost_limit = max(1, ghost_limit)
        self._ghost: deque[tuple[str, ...]] = deque()
        self._ghost_keys: set[tuple[str, ...]] = set()

    def label_for_insert(self, key: tuple[str, ...], *, seen: bool) -> str:
        if seen or key in self._ghost_keys:
            self._forget(key)
            return "am"
        return "a1"

    def label_for_hit(self, key: tuple[str, ...], label: str) -> str:
        if label == "a1":
            return "am"
        return label or "am"

    def note_evict(self, entry: CacheEntry) -> None:
        if entry.queue != "a1" or entry.key in self._ghost_keys:
            return
        self._ghost.append(entry.key)
        self._ghost_keys.add(entry.key)
        while len(self._ghost) > self.ghost_limit:
            self._ghost_keys.discard(self._ghost.popleft())

    def choose(self, entries: Sequence[CacheEntry]) -> CacheEntry | None:
        probation = [entry for entry in entries if entry.queue == "a1"]
        if probation:
            return min(probation, key=lambda entry: (entry.order, entry.last_used))
        if not entries:
            return None
        return min(entries, key=lambda entry: (entry.last_used, entry.order))

    def _forget(self, key: tuple[str, ...]) -> None:
        if key not in self._ghost_keys:
            return
        self._ghost_keys.discard(key)
        self._ghost = deque(item for item in self._ghost if item != key)


def policy_for(name: str) -> Eviction:
    """Resolve a configured eviction name. Add a policy by registering it here."""
    if name == "lowest-count":
        name = "lfu"
    policies: dict[str, type[Eviction]] = {
        "lru": LeastRecentlyUsed,
        "lfu": LeastFrequentlyUsed,
        "2q": TwoQueue,
    }
    try:
        return policies[name]()
    except KeyError as exc:
        known = ", ".join(sorted(policies))
        raise ValueError(f"Unknown eviction policy {name!r}. Known policies: {known}.") from exc
