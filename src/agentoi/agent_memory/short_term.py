"""Short-term prefix trie for one ontology."""

from __future__ import annotations

from agentoi.agent_memory.path import QueryPath
from agentoi.agent_memory.trie import PrefixTrie, TrieNode


class ShortTermMemory:
    """Trie of recent concept paths, capped by evicting rare leaves."""

    def __init__(self, max_nodes: int = 512) -> None:
        self.max_nodes = max(1, max_nodes)
        self.trie = PrefixTrie()

    def insert(self, path: QueryPath, answer: str) -> TrieNode | None:
        node = self.trie.insert(path, answer)
        self.evict_to_cap()
        if node is None:
            return None
        if node.concept_ids and self.trie.answer_for(node.concept_ids) is None:
            return None
        return node

    def answer_for(self, concept_ids: tuple[str, ...]) -> str | None:
        return self.trie.answer_for(concept_ids)

    def prefix_nodes(self, concept_ids: tuple[str, ...]) -> tuple[str, ...]:
        return self.trie.prefix_nodes(concept_ids)

    def evict_to_cap(self) -> list[tuple[str, ...]]:
        removed: list[tuple[str, ...]] = []
        while self.trie.node_count() > self.max_nodes:
            leaf = self.trie.lowest_leaf()
            if leaf is None:
                break
            removed.append(leaf.concept_ids)
            self.trie.detach(leaf)
        return removed

    def to_dict(self) -> dict[str, object]:
        return self.trie.to_dict()

    def load(self, payload: dict[str, object]) -> None:
        self.trie = PrefixTrie.from_dict(payload)
