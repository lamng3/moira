from dataclasses import replace
from pathlib import Path

import pytest

from agentoi.memory.cache.hot import HotCache
from agentoi.memory.cache.long_term import LongTermMemory
from agentoi.memory.cache.short_term import ShortTermMemory
from agentoi.memory.config import MemoryConfig
from agentoi.memory.eviction import LeastFrequentlyUsed, LeastRecentlyUsed, TwoQueue, policy_for
from agentoi.memory.path import QueryPath
from agentoi.memory.service import AgentMemory
from agentoi.memory.trie import TrieNode


def test_a_read_increments_frequency_and_leaves_the_promotion_count() -> None:
    short = ShortTermMemory(max_nodes=8, policy=policy_for("lfu"))
    short.insert(QueryPath.from_steps("organ systems", ["organ"]), "Digestive system.")
    leaf = short.trie.walk(("organ",))
    assert leaf.count == 1
    assert leaf.frequency == 1

    assert short.answer_for(("organ",)) == "Digestive system."

    assert leaf.count == 1
    assert leaf.frequency == 2

    long = LongTermMemory(max_entries=4, policy=policy_for("lfu"))
    long.promote(("organ",), "sparql", 3, "Digestive system.")
    assert long.lookup(("organ",)) == "Digestive system."
    assert long.patterns[("organ",)]["count"] == 3
    assert long.patterns[("organ",)]["frequency"] == 2


def test_lfu_keeps_a_question_that_was_read_often() -> None:
    cache = HotCache(capacity=2, policy=policy_for("lfu"))
    cache.put("once", "stored once")
    cache.put("often", "read often")
    cache.get("often")
    cache.get("often")
    cache.put("new", "just arrived")

    assert cache.get("often") == "read often"
    assert cache.get("once") is None


def test_lfu_treats_a_missing_frequency_as_the_stored_count() -> None:
    cache = HotCache(capacity=2, policy=policy_for("lfu"))
    cache.load(
        [
            {"question": "once", "answer": "stored once", "count": 1, "order": 1, "last_used": 1},
            {"question": "often", "answer": "read often", "count": 5, "order": 2, "last_used": 2},
        ]
    )
    cache.put("new", "just arrived")

    assert cache.get("often") == "read often"
    assert cache.get("once") is None
    assert TrieNode.from_dict({"concept_id": "organ", "count": 4, "answer": "saved"}).frequency == 4


def test_2q_evicts_a_one_time_question_before_a_repeated_one() -> None:
    cache = HotCache(capacity=2, policy=policy_for("2q"))
    cache.put("one-time", "seen once")
    cache.put("repeated", "seen again")
    cache.get("repeated")
    cache.put("fresh", "another one-time question")

    assert cache.get("repeated") == "seen again"
    assert cache.get("one-time") is None


def test_2q_keeps_a_ghost_hit_when_one_time_questions_arrive() -> None:
    cache = HotCache(capacity=2, policy=policy_for("2q"))
    cache.put("gone", "comes back")
    cache.put("other", "filler")
    cache.put("third", "pushes gone out")
    cache.put("gone", "comes back")
    cache.put("noise", "one time")
    cache.put("more noise", "also one time")

    saved = {row["question"]: row for row in cache.to_list()}
    assert saved["gone"]["queue"] == "am"
    assert all("ghost" not in row for row in saved.values())
    assert cache.get("gone") == "comes back"


def test_unknown_eviction_names_lfu_and_2q() -> None:
    with pytest.raises(ValueError, match="lfu") as raised:
        policy_for("nope")

    message = str(raised.value)
    assert "2q" in message
    assert "lru" in message


def test_default_eviction_is_lfu(tmp_path: Path) -> None:
    memory = AgentMemory(MemoryConfig(directory=tmp_path))
    lru = AgentMemory(replace(MemoryConfig(directory=tmp_path / "lru"), eviction="lru"))
    renamed = AgentMemory(replace(MemoryConfig(directory=tmp_path / "renamed"), eviction="lowest-count"))
    two_q = AgentMemory(replace(MemoryConfig(directory=tmp_path / "2q"), eviction="2q"))

    assert isinstance(memory.hot.policy, LeastFrequentlyUsed)
    assert isinstance(memory.short.policy, LeastFrequentlyUsed)
    assert isinstance(memory.long.policy, LeastFrequentlyUsed)
    assert isinstance(renamed.hot.policy, LeastFrequentlyUsed)
    assert isinstance(lru.hot.policy, LeastRecentlyUsed)
    assert isinstance(two_q.long.policy, TwoQueue)
