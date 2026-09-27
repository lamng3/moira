"""Short-term prefix trie for one ontology."""

from __future__ import annotations

from agentoi.memory.eviction import CacheEntry, EvictionPolicy, LowestCount
from agentoi.memory.trie import PrefixTrie


class ShortTermMemory:
    """Trie of recent concept paths, capped by the configured eviction policy."""

    def __init__(self, max_nodes: int = 512, policy: EvictionPolicy | None = None) -> None:
        self.max_nodes = max(1, max_nodes)
        self.policy = policy or LowestCount()
        self.trie = PrefixTrie()

    def answer_for(self, concept_ids: tuple[str, ...]) -> str | None:
        return self.trie.answer_for(concept_ids)

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
                    )
                    for leaf in leaves
                ]
            )
            if chosen is None:
                break
            leaf = next(item for item in leaves if item.concept_ids == chosen.key)
            removed.append(leaf.concept_ids)
            self.trie.detach(leaf)
        return removed

    def to_dict(self) -> dict[str, object]:
        return self.trie.to_dict()

    def load(self, payload: dict[str, object]) -> None:
        self.trie = PrefixTrie.from_dict(payload)
