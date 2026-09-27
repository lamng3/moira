"""Frequent concept paths kept after the short-term trie flushes them."""

from __future__ import annotations


class LongTermMemory:
    """Completed paths promoted out of the short-term trie."""

    def __init__(self) -> None:
        self.patterns: dict[tuple[str, ...], dict[str, object]] = {}

    def promote(
        self,
        concept_ids: tuple[str, ...],
        sparql: str,
        count: int,
        answer: str,
    ) -> None:
        if not concept_ids or not answer:
            return
        self.patterns[concept_ids] = {
            "concept_ids": list(concept_ids),
            "sparql": sparql,
            "count": count,
            "answer": answer,
        }

    def lookup(self, concept_ids: tuple[str, ...]) -> str | None:
        pattern = self.patterns.get(tuple(concept_ids))
        if pattern is None:
            return None
        answer = pattern.get("answer")
        return answer if isinstance(answer, str) and answer else None

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
            self.patterns[key] = {
                "concept_ids": list(key),
                "sparql": str(pattern.get("sparql") or ""),
                "count": int(pattern.get("count") or 0),
                "answer": answer,
            }
