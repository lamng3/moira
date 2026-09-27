"""Hot, short-term, and long-term query memory for one ontology."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from agentoi.agent_memory.hot import HotCache
from agentoi.agent_memory.long_term import LongTermMemory
from agentoi.agent_memory.path import QueryPath
from agentoi.agent_memory.short_term import ShortTermMemory
from agentoi.agent_memory.trie import TrieNode

DEFAULT_ROOT = Path("results/query_memory")
PROMOTE_AT = 3


def query_memory_dir(ontology: Path, root: Path | None = None) -> Path:
    """Directory for one ontology's query memory."""
    digest = hashlib.sha256(ontology.read_bytes()).hexdigest()[:16]
    return (root or DEFAULT_ROOT) / digest


@dataclass(frozen=True, slots=True)
class MemoryDecision:
    """What query memory can answer before the model runs."""

    answer: str | None = None
    prefix: tuple[str, ...] = ()
    source: str = "miss"


class AgentMemory:
    """Remember concept paths and reuse exact or full-path answers."""

    def __init__(
        self,
        directory: Path,
        *,
        promote_at: int = PROMOTE_AT,
        max_nodes: int = 512,
        hot_size: int = 32,
    ) -> None:
        self.directory = directory
        self.promote_at = promote_at
        self.hot = HotCache(hot_size)
        self.short = ShortTermMemory(max_nodes)
        self.long = LongTermMemory()
        self.load()

    def lookup_question(self, question: str) -> str | None:
        return self.hot.get(question)

    def lookup_path(self, path: QueryPath) -> str | None:
        if not path.concept_ids:
            return None
        answer = self.short.answer_for(path.concept_ids)
        if answer:
            return answer
        return self.long.lookup(path.concept_ids)

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
        prefix = self.prefix_nodes(path.concept_ids)
        if prefix and len(prefix) < len(path.concept_ids):
            return MemoryDecision(prefix=prefix, source="prefix")
        return MemoryDecision()

    def remember(self, path: QueryPath, answer: str) -> TrieNode | None:
        if not answer.strip():
            return None
        self.hot.put(path.question, answer)
        node = None
        if path.concept_ids:
            node = self.short.trie.insert(path, answer)
            if node is not None and node.answer and node.count >= self.promote_at:
                self.long.promote(node.concept_ids, node.sparql, node.count, node.answer)
            self.short.evict_to_cap()
        self.save()
        return node

    def save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        short_payload = {
            "version": 1,
            "hot": self.hot.to_list(),
            "trie": self.short.to_dict(),
        }
        long_payload = {"version": 1, "patterns": self.long.to_list()}
        (self.directory / "short_term.json").write_text(
            json.dumps(short_payload, indent=2),
            encoding="utf-8",
        )
        (self.directory / "long_term.json").write_text(
            json.dumps(long_payload, indent=2),
            encoding="utf-8",
        )

    def load(self) -> None:
        short_path = self.directory / "short_term.json"
        long_path = self.directory / "long_term.json"
        if short_path.is_file():
            payload = json.loads(short_path.read_text(encoding="utf-8"))
            hot = payload.get("hot")
            trie = payload.get("trie")
            if isinstance(hot, list):
                self.hot.load(hot)
            if isinstance(trie, dict):
                self.short.load(trie)
        if long_path.is_file():
            payload = json.loads(long_path.read_text(encoding="utf-8"))
            patterns = payload.get("patterns")
            if isinstance(patterns, list):
                self.long.load(patterns)


def cached_prefix_note(labels: list[str]) -> str:
    """Prompt lines for concepts already seen on a shared prefix."""
    if not labels:
        return ""
    lines = ["ALREADY SEEN CONCEPTS:"]
    lines.extend(f"- {label}" for label in labels)
    return "\n".join(lines)
