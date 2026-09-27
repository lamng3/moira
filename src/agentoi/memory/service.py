"""Query cache for one ontology."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agentoi.memory.cache.hot import HotCache
from agentoi.memory.cache.long_term import LongTermMemory
from agentoi.memory.cache.short_term import ShortTermMemory
from agentoi.memory.config import MemoryConfig
from agentoi.memory.eviction import LeastRecentlyUsed, policy_for
from agentoi.memory.path import QueryPath
from agentoi.memory.replay import GraphReplay
from agentoi.memory.trie import TrieNode


@dataclass(frozen=True, slots=True)
class MemoryDecision:
    """What the cache can answer before the model runs."""

    answer: str | None = None
    prefix: tuple[str, ...] = ()
    source: str = "miss"


class AgentMemory:
    """Hot hash, short-term trie, and long-term hash for concept paths."""

    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self.directory = config.directory
        tier_policy = policy_for(config.eviction)
        self.hot = HotCache(config.hot_entries, LeastRecentlyUsed())
        self.short = ShortTermMemory(config.short_term_nodes, tier_policy)
        self.long = LongTermMemory(config.long_term_entries, tier_policy)
        self.replay = GraphReplay(self.short, self.long, promote_at=config.promote_at)
        self.load()

    def lookup_question(self, question: str) -> str | None:
        return self.hot.get(question)

    def lookup_path(self, path: QueryPath) -> str | None:
        return self.replay.replay(path)

    def prefix_nodes(self, concept_ids: tuple[str, ...]) -> tuple[str, ...]:
        return self.short.prefix_nodes(concept_ids)

    def consult_question(self, question: str) -> MemoryDecision:
        answer = self.lookup_question(question)
        if answer:
            return MemoryDecision(answer=answer, source="hot")
        return MemoryDecision()

    def consult_path(self, path: QueryPath) -> MemoryDecision:
        answer = self.lookup_path(path)
        if answer:
            return MemoryDecision(answer=answer, source="path")
        prefix = self.replay.replay_prefix(path)
        if prefix:
            return MemoryDecision(prefix=prefix, source="prefix")
        return MemoryDecision()

    def remember(self, path: QueryPath, answer: str) -> TrieNode | None:
        if not answer.strip():
            return None
        self.hot.put(path.question, answer)
        node = self.replay.capture(path, answer)
        self.save()
        return node

    def save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        _write(self.directory / "hot.json", {"version": 1, "entries": self.hot.to_list()})
        _write(self.directory / "short_term.json", {"version": 1, "trie": self.short.to_dict()})
        _write(self.directory / "long_term.json", {"version": 1, "patterns": self.long.to_list()})

    def load(self) -> None:
        hot = _read(self.directory / "hot.json")
        short = _read(self.directory / "short_term.json")
        long = _read(self.directory / "long_term.json")
        if isinstance(hot, dict) and isinstance(hot.get("entries"), list):
            self.hot.load(hot["entries"])
        if isinstance(short, dict) and isinstance(short.get("trie"), dict):
            self.short.load(short["trie"])
        if isinstance(long, dict) and isinstance(long.get("patterns"), list):
            self.long.load(long["patterns"])


def cached_prefix_note(labels: list[str]) -> str:
    """Prompt lines for concepts already seen on a shared prefix."""
    if not labels:
        return ""
    lines = ["ALREADY SEEN CONCEPTS:"]
    lines.extend(f"- {label}" for label in labels)
    return "\n".join(lines)


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read(path: Path) -> object | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
