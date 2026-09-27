from dataclasses import replace
from pathlib import Path

from agentoi.memory.cache.hot import HotCache
from agentoi.memory.config import MemoryConfig
from agentoi.memory.path import SUBCLASS_OF, QueryPath
from agentoi.memory.service import AgentMemory
from agentoi.memory.trie import PrefixTrie


def _memory(directory: Path, **overrides: object) -> AgentMemory:
    config = MemoryConfig(directory=directory, promote_at=3, short_term_nodes=32, hot_entries=4)
    return AgentMemory(replace(config, **overrides))


def test_trie_counts_a_shared_prefix_and_stores_the_sparql_pattern() -> None:
    trie = PrefixTrie()
    organ = QueryPath.from_steps(
        "what organ systems are included?",
        ["http://example.org/organ", "digestive"],
        [("http://example.org/organ", SUBCLASS_OF, "digestive")],
    )
    blood = QueryPath.from_steps("what is blood?", ["http://example.org/organ", "blood"])

    trie.insert(organ, "The digestive system is included.")
    trie.insert(blood, "Blood is a tissue.")

    shared = trie.walk(("http://example.org/organ",))
    assert shared.count == 2
    assert trie.walk(("http://example.org/organ", "digestive")).count == 1
    assert "?c0 a <http://example.org/organ> ." in organ.sparql
    assert f"?c0 <{SUBCLASS_OF}> ?c1 ." in organ.sparql
    assert trie.answer_for(("http://example.org/organ",)) is None
    assert trie.answer_for(organ.concept_ids) == "The digestive system is included."


def test_hot_cache_returns_an_exact_question_and_forgets_the_least_recent() -> None:
    cache = HotCache(capacity=2)
    cache.put("What are the organ systems included?", "Digestive system.")
    cache.put("Where is the heart?", "In the chest.")

    assert cache.get("  what are the organ systems included? ") == "Digestive system."
    cache.put("Where is the brain?", "In the head.")

    assert cache.get("Where is the heart?") is None
    assert cache.get("Where is the brain?") == "In the head."


def test_full_path_reuses_the_answer_for_a_different_question(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    first = QueryPath.from_steps("What are the organ systems included?", ["organ", "digestive"])
    memory.remember(first, "The digestive system is included.")
    paraphrased = QueryPath.from_steps(
        "Which systems does this anatomy contain?",
        ["organ", "digestive"],
    )

    assert memory.lookup_question(paraphrased.question) is None
    assert memory.lookup_path(paraphrased) == "The digestive system is included."
    assert memory.consult_path(paraphrased).source == "path"


def test_prefix_match_returns_nodes_without_an_answer(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    memory.remember(
        QueryPath.from_steps("organ systems", ["organ", "digestive"]),
        "The digestive system is included.",
    )
    related = QueryPath.from_steps("what is blood?", ["organ", "blood"])

    decision = memory.consult_path(related)

    assert decision.answer is None
    assert decision.source == "prefix"
    assert decision.prefix == ("organ",)
    assert memory.lookup_path(related) is None


def test_flush_promotes_a_repeated_path_and_drops_a_one_off(tmp_path: Path) -> None:
    memory = _memory(tmp_path, short_term_nodes=5, hot_entries=8)
    repeated = QueryPath.from_steps("what organ systems are included?", ["organ", "digestive"])
    for _ in range(3):
        memory.remember(repeated, "The digestive system is included.")

    for name in ("bone", "skin", "hair"):
        memory.remember(QueryPath.from_steps(name, [name]), f"About {name}.")

    assert memory.short.answer_for(("bone",)) is None
    assert memory.lookup_path(repeated) == "The digestive system is included."
    assert memory.long.lookup(("organ", "digestive")) == "The digestive system is included."
    assert (tmp_path / "hot.json").is_file()
    assert (tmp_path / "short_term.json").is_file()
    assert (tmp_path / "long_term.json").is_file()

    restored = _memory(tmp_path, short_term_nodes=5, hot_entries=8)
    assert restored.lookup_question(repeated.question) == "The digestive system is included."
    assert restored.replay.replay(repeated) == "The digestive system is included."
    assert "digestive" in restored.long.patterns[("organ", "digestive")]["sparql"]


def test_replay_returns_a_captured_path_without_the_model(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    captured = QueryPath.from_steps("organ systems", ["organ", "digestive"])
    memory.remember(captured, "The digestive system is included.")
    same_path = QueryPath.from_steps("a different question", ["organ", "digestive"])

    assert memory.replay.replay(same_path) == "The digestive system is included."
    assert memory.replay.replay_prefix(
        QueryPath.from_steps("what is blood?", ["organ", "blood"])
    ) == ("organ",)


def test_long_term_cap_drops_the_rarer_pattern(tmp_path: Path) -> None:
    memory = _memory(tmp_path, promote_at=1, long_term_entries=1, short_term_nodes=32)
    frequent = QueryPath.from_steps("organ systems", ["organ", "digestive"])
    rare = QueryPath.from_steps("a bone", ["bone"])
    for _ in range(3):
        memory.remember(frequent, "The digestive system is included.")
    memory.remember(rare, "About bone.")

    assert ("bone",) not in memory.long.patterns
    assert ("organ", "digestive") in memory.long.patterns
    assert memory.replay.replay(frequent) == "The digestive system is included."
