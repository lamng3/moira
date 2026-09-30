"""Checksum of word shingles for a remembered answer.

A bloom filter would say whether a token was seen. This checksum says which
stored answer a definition belongs to, so the check can name that concept.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from agentoi.memory.path import normalize_question
from agentoi.memory.trie import TrieNode
from agentoi.memory.view import short_label

MIN_SHINGLES = 4
MIN_SCORE = 0.75
MIN_MARGIN = 0.15
_STOP = frozenset({"what", "is", "a", "the", "of", "that", "and"})
_WORD = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


@dataclass(frozen=True, slots=True)
class AnswerSignature:
    """Hashes of one answer and the concept path that produced it."""

    hashes: frozenset[int]
    words: tuple[str, ...]
    concept_ids: tuple[str, ...]


class AnswerIndex:
    """Small set of answer checksums. The hot cache stays the question key."""

    def __init__(self) -> None:
        self._rows: list[AnswerSignature] = []

    def clear(self) -> None:
        self._rows.clear()

    def add(self, answer: str, concept_ids: tuple[str, ...]) -> None:
        if not answer.strip() or not concept_ids:
            return
        words = tuple(_words(answer))
        hashes = _hashes_from_words(words)
        if len(hashes) < MIN_SHINGLES:
            return
        key = tuple(concept_ids)
        self._rows = [row for row in self._rows if row.concept_ids != key]
        self._rows.append(AnswerSignature(hashes, words, key))

    def match(self, question: str) -> str | None:
        """Return the concept label when the question is mostly one stored answer."""
        query_words = _words(question)
        query = _hashes_from_words(query_words)
        if len(query) < MIN_SHINGLES or not self._rows:
            return None
        scored = sorted(
            ((_overlap(query, row.hashes), row) for row in self._rows),
            key=lambda item: item[0],
            reverse=True,
        )
        best, winner = scored[0]
        second = scored[1][0] if len(scored) > 1 else 0.0
        if best < MIN_SCORE or best - second < MIN_MARGIN:
            scored = sorted(
                ((_tolerant_overlap(query_words, row), row) for row in self._rows),
                key=lambda item: item[0],
                reverse=True,
            )
            best, winner = scored[0]
            second = scored[1][0] if len(scored) > 1 else 0.0
            if best < MIN_SCORE or best - second < MIN_MARGIN:
                return None
        return short_label(winner.concept_ids[-1])


def shingle_hashes(text: str) -> frozenset[int]:
    """Stable checksum of every three-word window in the text."""
    return _hashes_from_words(_words(text))


def rebuild_index(
    index: AnswerIndex,
    root: TrieNode,
    patterns: list[dict[str, object]],
) -> None:
    """Fill the checksums from the concept trie and the long-term patterns."""
    index.clear()
    for node in _walk(root):
        if node.answer and node.concept_ids:
            index.add(node.answer, node.concept_ids)
    for pattern in patterns:
        concept_ids = pattern.get("concept_ids")
        answer = pattern.get("answer")
        if isinstance(concept_ids, list) and isinstance(answer, str):
            index.add(answer, tuple(str(item) for item in concept_ids))


def _words(text: str) -> list[str]:
    folded = normalize_question(text).replace("relevant concept clusters", " ")
    return [word for word in _WORD.findall(folded) if word not in _STOP]


def _hashes_from_words(words: list[str] | tuple[str, ...]) -> frozenset[int]:
    return frozenset(_hash(" ".join(words[index : index + 3])) for index in range(len(words) - 2))


def _overlap(query: frozenset[int], answer: frozenset[int]) -> float:
    if not query:
        return 0.0
    return len(query & answer) / len(query)


def _tolerant_overlap(query_words: list[str], row: AnswerSignature) -> float:
    """Rescore after correcting one misspelled word of length 5 or more."""
    corrected = _correct_one(query_words, row.words)
    if corrected is None:
        return _overlap(_hashes_from_words(query_words), row.hashes)
    return _overlap(_hashes_from_words(corrected), row.hashes)


def _correct_one(query_words: list[str], answer_words: tuple[str, ...]) -> list[str] | None:
    pool = set(answer_words)
    missing = [index for index, word in enumerate(query_words) if word not in pool]
    if len(missing) != 1:
        return None
    index = missing[0]
    word = query_words[index]
    if len(word) < 5:
        return None
    neighbors = [item for item in pool if _edit_distance(word, item) == 1]
    if len(neighbors) != 1:
        return None
    corrected = list(query_words)
    corrected[index] = neighbors[0]
    return corrected


def _edit_distance(left: str, right: str) -> int:
    if abs(len(left) - len(right)) > 1:
        return 2
    previous = list(range(len(right) + 1))
    for left_char in left:
        current = [previous[0] + 1]
        row_min = current[0]
        for index, right_char in enumerate(right, 1):
            cost = 0 if left_char == right_char else 1
            current.append(min(current[-1] + 1, previous[index] + 1, previous[index - 1] + cost))
            row_min = min(row_min, current[-1])
        if row_min > 1:
            return 2
        previous = current
    return previous[-1]


def _hash(text: str) -> int:
    return int.from_bytes(hashlib.blake2s(text.encode(), digest_size=8).digest(), "big")


def _walk(node: TrieNode):
    yield node
    for child in node.children.values():
        yield from _walk(child)
