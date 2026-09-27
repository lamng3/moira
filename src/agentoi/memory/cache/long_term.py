"""Frequent concept paths kept on disk after the short-term trie promotes them."""

from __future__ import annotations

from agentoi.memory.eviction import CacheEntry, EvictionPolicy, LowestCount


class LongTermMemory:
    """Persistent hash of completed paths, capped by the eviction policy."""

    def __init__(self, max_entries: int = 1024, policy: EvictionPolicy | None = None) -> None:
        self.max_entries = max(1, max_entries)
        self.policy = policy or LowestCount()
        self.patterns: dict[tuple[str, ...], dict[str, object]] = {}
        self._clock = 1

    def promote(
        self,
        concept_ids: tuple[str, ...],
        sparql: str,
        count: int,
        answer: str,
    ) -> None:
        if not concept_ids or not answer:
            return
        self._clock += 1
        self.patterns[concept_ids] = {
            "concept_ids": list(concept_ids),
            "sparql": sparql,
            "count": count,
            "answer": answer,
            "order": self._clock,
            "last_used": self._clock,
        }
        self.evict_to_cap()

    def lookup(self, concept_ids: tuple[str, ...]) -> str | None:
        pattern = self.patterns.get(tuple(concept_ids))
        if pattern is None:
            return None
        self._clock += 1
        pattern["last_used"] = self._clock
        answer = pattern.get("answer")
        return answer if isinstance(answer, str) and answer else None

    def evict_to_cap(self) -> None:
        while len(self.patterns) > self.max_entries:
            chosen = self.policy.choose(
                [
                    CacheEntry(
                        key=key,
                        count=int(pattern.get("count") or 0),
                        order=int(pattern.get("order") or 0),
                        last_used=int(pattern.get("last_used") or 0),
                    )
                    for key, pattern in self.patterns.items()
                ]
            )
            if chosen is None:
                return
            self.patterns.pop(chosen.key, None)

    def to_list(self) -> list[dict[str, object]]:
        return list(self.patterns.values())

    def load(self, patterns: list[object]) -> None:
        self.patterns.clear()
        for pattern in patterns:
            if not isinstance(pattern, dict):
                continue
            concept_ids = pattern.get("concept_ids")
            answer = pattern.get("answer")
            if not isinstance(concept_ids, list) or not isinstance(answer, str):
                continue
            key = tuple(str(item) for item in concept_ids)
            order = int(pattern.get("order") or self._clock)
            last_used = int(pattern.get("last_used") or order)
            self.patterns[key] = {
                "concept_ids": list(key),
                "sparql": str(pattern.get("sparql") or ""),
                "count": int(pattern.get("count") or 0),
                "answer": answer,
                "order": order,
                "last_used": last_used,
            }
            self._clock = max(self._clock, order + 1, last_used + 1)
        self.evict_to_cap()
