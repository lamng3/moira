"""Exact-question answers kept in front of retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from moira.memory.eviction import CacheEntry, Eviction, LeastRecentlyUsed
from moira.memory.path import normalize_question


@dataclass
class _HotItem:
    answer: str
    count: int
    order: int
    last_used: int
    frequency: int = 1
    queue: str = ""


class HotCache:
    """Small exact-question cache. The default policy is least recently used."""

    def __init__(self, capacity: int = 32, policy: Eviction | None = None) -> None:
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
        item.frequency += 1
        item.queue = self.policy.label_for_hit((key,), item.queue)
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
            frequency=(current.frequency + 1) if current else 1,
            queue=self.policy.label_for_insert((key,), seen=current is not None),
        )
        self._clock += 1
        self._evict()

    def _evict(self) -> None:
        while len(self._items) > self.capacity:
            chosen = self.policy.choose(
                [
                    CacheEntry(
                        key=(key,),
                        count=item.count,
                        order=item.order,
                        last_used=item.last_used,
                        frequency=item.frequency,
                        queue=item.queue,
                    )
                    for key, item in self._items.items()
                ]
            )
            if chosen is None:
                return
            removed = self._items.pop(chosen.key[0], None)
            if removed is not None:
                self.policy.note_evict(chosen)

    def to_list(self) -> list[dict[str, object]]:
        return [
            {
                "question": question,
                "answer": item.answer,
                "count": item.count,
                "frequency": item.frequency,
                "queue": item.queue,
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
            count = int(entry.get("count") or 1)
            frequency = entry.get("frequency")
            self._items[key] = _HotItem(
                answer=answer,
                count=count,
                order=order,
                last_used=last_used,
                frequency=int(frequency) if frequency is not None else count,
                queue=str(entry.get("queue") or ""),
            )
            self._clock = max(self._clock, order + 1, last_used + 1)
        self._evict()
