"""Replay a captured concept path instead of running it again."""

from __future__ import annotations

from agentoi.memory.cache.long_term import LongTermMemory
from agentoi.memory.cache.short_term import ShortTermMemory
from agentoi.memory.path import QueryPath
from agentoi.memory.trie import TrieNode


def _sorted_path(path: QueryPath) -> QueryPath:
    """Key a capture by concept id, ignoring retrieval order."""
    ordered = tuple(sorted(path.concept_ids))
    if ordered == path.concept_ids:
        return path
    return QueryPath.from_steps(path.question, ordered, path.edges)


class GraphReplay:
    """Capture a concept set once, then replay the stored answer.

    Replay reads the capture. It does not execute the SPARQL pattern.
    """

    def __init__(self, short: ShortTermMemory, long: LongTermMemory, *, promote_at: int) -> None:
        self.short = short
        self.long = long
        self.promote_at = promote_at

    def capture(self, path: QueryPath, answer: str) -> TrieNode | None:
        """Store a completed path and promote it when it is seen often enough."""
        if not path.concept_ids:
            return None
        node = self.short.trie.insert(_sorted_path(path), answer)
        if node is not None and node.answer and node.count >= self.promote_at:
            self.long.promote(node.concept_ids, node.sparql, node.count, node.answer)
        self.short.evict_to_cap()
        return node

    def replay(self, path: QueryPath) -> str | None:
        """Return the captured answer when the same concepts were stored."""
        if not path.concept_ids:
            return None
        concept_ids = _sorted_path(path).concept_ids
        answer = self.short.answer_for(concept_ids)
        if answer:
            return answer
        return self.long.lookup(concept_ids)

    def replay_prefix(self, path: QueryPath) -> tuple[str, ...]:
        """Return a shared prefix. A full-path match is left to replay()."""
        prefix = self.short.prefix_nodes(path.concept_ids)
        if prefix and len(prefix) < len(path.concept_ids):
            return prefix
        return ()
