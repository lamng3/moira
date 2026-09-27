"""Exact-question answers kept in front of retrieval."""

from __future__ import annotations

from collections import OrderedDict

from agentoi.agent_memory.path import normalize_question


class HotCache:
    """Small LRU of normalized questions and their answers."""

    def __init__(self, capacity: int = 32) -> None:
        self.capacity = max(1, capacity)
        self._items: OrderedDict[str, str] = OrderedDict()

    def get(self, question: str) -> str | None:
        key = normalize_question(question)
        answer = self._items.get(key)
        if answer is None:
            return None
        self._items.move_to_end(key)
        return answer

    def put(self, question: str, answer: str) -> None:
        key = normalize_question(question)
        if not key or not answer:
            return
        self._items[key] = answer
        self._items.move_to_end(key)
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)

    def to_list(self) -> list[dict[str, str]]:
        return [{"question": question, "answer": answer} for question, answer in self._items.items()]

    def load(self, entries: list[object]) -> None:
        self._items.clear()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            question = entry.get("question")
            answer = entry.get("answer")
            if isinstance(question, str) and isinstance(answer, str):
                self.put(question, answer)
