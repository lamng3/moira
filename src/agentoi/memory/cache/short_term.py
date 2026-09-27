"""Short-term prefix trie for one ontology."""

from __future__ import annotations

from agentoi.memory.eviction import CacheEntry, Eviction, LeastFrequentlyUsed
from agentoi.memory.path import QueryPath
from agentoi.memory.trie import PrefixTrie, TrieNode


class ShortTermMemory:
    """Trie of recent concept paths, capped by the configured eviction policy."""

    def __init__(self, max_nodes: int = 512, policy: Eviction | None = None) -> None:
        self.max_nodes = max(1, max_nodes)
        self.policy = policy or LeastFrequentlyUsed()
        self.trie = PrefixTrie()

    def insert(self, path: QueryPath, answer: str) -> TrieNode | None:
        node = self.trie.insert(path, answer)
        if node is None:
            return None
        node.queue = self.policy.label_for_insert(node.concept_ids, seen=node.count > 1)
        if node.frequency == 0:
            node.frequency = node.count if node.count > 1 else 1
        else:
            node.frequency += 1
        return node

    def answer_for(self, concept_ids: tuple[str, ...]) -> str | None:
        node = self.trie.walk(concept_ids)
        if node.concept_ids != tuple(concept_ids) or not node.answer:
            return None
        if node.frequency == 0:
            node.frequency = node.count
        node.frequency += 1
        node.queue = self.policy.label_for_hit(node.concept_ids, node.queue)
        self.trie.note_read(node)
        return node.answer

    def prefix_nodes(self, concept_ids: tuple[str, ...]) -> tuple[str, ...]:
        return self.trie.prefix_nodes(concept_ids)

    def evict_to_cap(self) -> list[tuple[str, ...]]:
        removed: list[tuple[str, ...]] = []
        while self.trie.node_count() > self.max_nodes:
            leaves = self.trie.leaves()
            chosen = self.policy.choose(
                [
                    CacheEntry(
                        key=leaf.concept_ids,
                        count=leaf.count,
                        order=leaf.order,
                        last_used=leaf.last_used,
                        frequency=leaf.frequency or leaf.count,
                        queue=leaf.queue,
                    )
                    for leaf in leaves
                ]
            )
            if chosen is None:
                break
            leaf = next(item for item in leaves if item.concept_ids == chosen.key)
            removed.append(leaf.concept_ids)
            self.policy.note_evict(chosen)
            self.trie.detach(leaf)
        return removed

    def to_dict(self) -> dict[str, object]:
        return self.trie.to_dict()

    def load(self, payload: dict[str, object]) -> None:
        self.trie = PrefixTrie.from_dict(payload)
