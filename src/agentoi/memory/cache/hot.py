"""Exact-question answers kept in front of retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from agentoi.memory.eviction import CacheEntry, EvictionPolicy, LeastRecentlyUsed
from agentoi.memory.path import normalize_question


@dataclass
class _HotItem:
    answer: str
    count: int
    order: int
    last_used: int


class HotCache:
    """Small exact-question cache. The default policy is least recently used."""

    def __init__(self, capacity: int = 32, policy: EvictionPolicy | None = None) -> None:
        self.capacity = max(1, capacity)
        self.policy = policy or LeastRecentlyUsed()
        self._items: dict[str, _HotItem] = {}
        self._clock = 1

    def get(self, question: str) -> str | None:
        key = normalize_question(question)
        item = self._items.get(key)
        if item is None:
            return None
        item.last_used = self._clock
        self._clock += 1
        return item.answer

    def put(self, question: str, answer: str) -> None:
        key = normalize_question(question)
        if not key or not answer:
            return
        current = self._items.get(key)
        self._items[key] = _HotItem(
            answer=answer,
            count=(current.count + 1) if current else 1,
            order=current.order if current else self._clock,
            last_used=self._clock,
        )
        self._clock += 1
        self._evict()

    def _evict(self) -> None:
        while len(self._items) > self.capacity:
            chosen = self.policy.choose(
                [
                    CacheEntry(key=(key,), count=item.count, order=item.order, last_used=item.last_used)
                    for key, item in self._items.items()
                ]
            )
            if chosen is None:
                return
            self._items.pop(chosen.key[0], None)

    def to_list(self) -> list[dict[str, object]]:
        return [
            {
                "question": question,
                "answer": item.answer,
                "count": item.count,
                "order": item.order,
                "last_used": item.last_used,
            }
            for question, item in self._items.items()
        ]

    def load(self, entries: list[object]) -> None:
        self._items.clear()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            question = entry.get("question")
            answer = entry.get("answer")
            if not isinstance(question, str) or not isinstance(answer, str):
                continue
            key = normalize_question(question)
            order = int(entry.get("order") or self._clock)
            last_used = int(entry.get("last_used") or order)
            self._items[key] = _HotItem(
                answer=answer,
                count=int(entry.get("count") or 1),
                order=order,
                last_used=last_used,
            )
            self._clock = max(self._clock, order + 1, last_used + 1)
        self._evict()
