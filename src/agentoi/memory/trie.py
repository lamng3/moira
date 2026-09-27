"""Prefix trie of concept paths."""

from __future__ import annotations

from dataclasses import dataclass, field

from agentoi.memory.path import QueryPath, render_sparql


@dataclass
class TrieNode:
    """One concept along a captured path."""

    concept_id: str | None = None
    concept_ids: tuple[str, ...] = ()
    count: int = 0
    sparql: str = ""
    answer: str | None = None
    order: int = 0
    last_used: int = 0
    frequency: int = 0
    queue: str = ""
    children: dict[str, TrieNode] = field(default_factory=dict)

    @property
    def is_root(self) -> bool:
        return self.concept_id is None

    def to_dict(self) -> dict[str, object]:
        return {
            "concept_id": self.concept_id,
            "concept_ids": list(self.concept_ids),
            "count": self.count,
            "sparql": self.sparql,
            "answer": self.answer,
            "order": self.order,
            "last_used": self.last_used,
            "frequency": self.frequency,
            "queue": self.queue,
            "children": {key: child.to_dict() for key, child in self.children.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> TrieNode:
        children_payload = payload.get("children") or {}
        children = {
            str(key): cls.from_dict(value)
            for key, value in children_payload.items()
            if isinstance(value, dict)
        }
        concept_ids = payload.get("concept_ids") or []
        concept_id = payload.get("concept_id")
        answer = payload.get("answer")
        count = int(payload.get("count") or 0)
        frequency = payload.get("frequency")
        return cls(
            concept_id=concept_id if isinstance(concept_id, str) else None,
            concept_ids=tuple(str(item) for item in concept_ids),
            count=count,
            sparql=str(payload.get("sparql") or ""),
            answer=answer if isinstance(answer, str) else None,
            order=int(payload.get("order") or 0),
            last_used=int(payload.get("last_used") or 0),
            frequency=int(frequency) if frequency is not None else count,
            queue=str(payload.get("queue") or ""),
            children=children,
        )


class PrefixTrie:
    """Concept-id trie. Answers live only on a completed path."""

    def __init__(self) -> None:
        self.root = TrieNode()
        self._order = 1
        self._clock = 1

    def insert(self, path: QueryPath, answer: str) -> TrieNode | None:
        """Record one completed path and return its terminal node."""
        if not path.concept_ids:
            return None
        node = self.root
        node.count += 1
        for index, concept_id in enumerate(path.concept_ids):
            child = node.children.get(concept_id)
            prefix = path.concept_ids[: index + 1]
            if child is None:
                child = TrieNode(concept_id=concept_id, concept_ids=prefix, order=self._order)
                self._order += 1
                node.children[concept_id] = child
            child.count += 1
            child.last_used = self._clock
            self._clock += 1
            child.sparql = render_sparql(prefix, path.edges)
            node = child
        node.answer = answer
        return node

    def walk(self, concept_ids: tuple[str, ...] | list[str]) -> TrieNode:
        """Return the deepest node that matches the head of this sequence."""
        node = self.root
        for concept_id in concept_ids:
            child = node.children.get(concept_id)
            if child is None:
                break
            node = child
        return node

    def answer_for(self, concept_ids: tuple[str, ...] | list[str]) -> str | None:
        """Return an answer only when the whole concept sequence was stored."""
        node = self.walk(concept_ids)
        if node.concept_ids != tuple(concept_ids) or not node.answer:
            return None
        return node.answer

    def prefix_nodes(self, concept_ids: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        """Return the concept ids of the longest stored prefix."""
        return self.walk(concept_ids).concept_ids

    def leaves(self) -> list[TrieNode]:
        return [node for node in _walk(self.root) if not node.is_root and not node.children]

    def node_count(self) -> int:
        return _count(self.root)

    def note_read(self, node: TrieNode) -> None:
        node.last_used = self._clock
        self._clock += 1

    def detach(self, leaf: TrieNode) -> None:
        parent = self.walk(leaf.concept_ids[:-1])
        if leaf.concept_id is not None:
            parent.children.pop(leaf.concept_id, None)

    def to_dict(self) -> dict[str, object]:
        return self.root.to_dict()

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> PrefixTrie:
        trie = cls()
        trie.root = TrieNode.from_dict(payload)
        nodes = list(_walk(trie.root))
        trie._order = max((node.order for node in nodes), default=0) + 1
        trie._clock = max((node.last_used for node in nodes), default=0) + 1
        return trie


def _walk(node: TrieNode):
    yield node
    for child in node.children.values():
        yield from _walk(child)


def _count(node: TrieNode) -> int:
    return 1 + sum(_count(child) for child in node.children.values())
