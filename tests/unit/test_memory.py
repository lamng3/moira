from dataclasses import replace
from pathlib import Path

from moira.memory.cache.hot import HotCache
from moira.memory.config import MemoryConfig
from moira.memory.path import SUBCLASS_OF, QueryPath
from moira.memory.service import AgentMemory
from moira.workspace import _converged_hot
from moira.memory.trie import PrefixTrie


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


def test_a_different_concept_set_does_not_replay(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    memory.remember(
        QueryPath.from_steps("organ systems", ["organ", "digestive"]),
        "The digestive system is included.",
    )
    related = QueryPath.from_steps("what is blood?", ["organ", "blood"])

    decision = memory.consult_path(related)

    assert decision.answer is None
    assert decision.source == "miss"
    assert decision.prefix == ()
    assert memory.lookup_path(related) is None


def test_same_concepts_in_another_order_replay(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    memory.remember(
        QueryPath.from_steps("what organ systems are part of the mouse?", ["organ", "digestive"]),
        "The digestive system is included.",
    )
    reordered = QueryPath.from_steps(
        "list the organ systems in the mouse",
        ["digestive", "organ"],
    )

    assert memory.lookup_question(reordered.question) is None
    assert memory.lookup_path(reordered) == "The digestive system is included."
    assert memory.consult_path(reordered).source == "path"


def test_flush_promotes_a_repeated_path_and_drops_a_one_off(tmp_path: Path) -> None:
    memory = _memory(tmp_path, short_term_nodes=5, hot_entries=8)
    repeated = QueryPath.from_steps("what organ systems are included?", ["organ", "digestive"])
    for _ in range(3):
        memory.remember(repeated, "The digestive system is included.")

    for name in ("bone", "skin", "hair"):
        memory.remember(QueryPath.from_steps(name, [name]), f"About {name}.")

    assert memory.short.answer_for(("bone",)) is None
    assert memory.lookup_path(repeated) == "The digestive system is included."
    assert memory.long.lookup(("digestive", "organ")) == "The digestive system is included."
    assert (tmp_path / "hot.json").is_file()
    assert (tmp_path / "short_term.json").is_file()
    assert (tmp_path / "long_term.json").is_file()

    restored = _memory(tmp_path, short_term_nodes=5, hot_entries=8)
    assert restored.lookup_question(repeated.question) == "The digestive system is included."
    assert restored.replay.replay(repeated) == "The digestive system is included."
    assert "digestive" in restored.long.patterns[("digestive", "organ")]["sparql"]


def test_the_same_content_words_share_one_hot_answer(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    short = (
        'The organ systems in the mouse include:\n\n'
        'The most relevant concepts that answer the question are "organ system" and "visceral organ system".'
    )
    full = (
        "The organ systems in the mouse include:\n\n"
        "* Visceral organ system\n"
        "* Digestive system (which is a part of the visceral organ system)\n\n"
        'The relevant concepts used to answer this question include "organ system" and "digestive system".'
    )
    memory.hot.put("What are the organ systems in the mouse?", short)
    memory.hot.put("list the organ systems in the mouse", full)
    memory.hot.put("What is the heart?", "A heart is an organ.")
    memory.hot.put("Where is the heart?", "In the chest.")

    listed = _converged_hot(memory, "list the organ systems in the mouse")
    asked = _converged_hot(memory, "What are the organ systems in the mouse?")

    assert listed == asked
    assert "Visceral organ system" in listed
    assert "Digestive system" in listed
    assert "relevant concepts" not in listed.lower()
    assert _converged_hot(memory, "Where is the heart?") == "In the chest."
    assert _converged_hot(memory, "What is the heart?") == "A heart is an organ."


def test_a_plural_wording_reuses_the_stored_model_answer(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    answer = "The digestive system of a mouse consists of the esophagus and the stomach."
    memory.hot.put("describe for me the digestive system of a mouse", answer)

    assert _converged_hot(memory, "describe for me digestive systems of a mouse") == answer
    assert _converged_hot(memory, "where is the digestive system of a mouse") == ""


def test_replay_returns_a_captured_path_without_the_model(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    captured = QueryPath.from_steps("organ systems", ["organ", "digestive"])
    memory.remember(captured, "The digestive system is included.")
    same_path = QueryPath.from_steps("a different question", ["organ", "digestive"])

    assert memory.replay.replay(same_path) == "The digestive system is included."
    assert memory.replay.replay(
        QueryPath.from_steps("list the organ systems", ["digestive", "organ"])
    ) == "The digestive system is included."


def test_long_term_cap_drops_the_rarer_pattern(tmp_path: Path) -> None:
    memory = _memory(tmp_path, promote_at=1, long_term_entries=1, short_term_nodes=32)
    frequent = QueryPath.from_steps("organ systems", ["organ", "digestive"])
    rare = QueryPath.from_steps("a bone", ["bone"])
    for _ in range(3):
        memory.remember(frequent, "The digestive system is included.")
    memory.remember(rare, "About bone.")

    assert ("bone",) not in memory.long.patterns
    assert ("digestive", "organ") in memory.long.patterns
    assert memory.replay.replay(frequent) == "The digestive system is included."
