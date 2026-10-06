"""Deterministic web-search providers for offline runs."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import TYPE_CHECKING, Mapping

from .base import WebSearchProvider

if TYPE_CHECKING:
    from moira.algorithms.graph import ConceptGraph


class StaticSearchProvider(WebSearchProvider):
    """Return configured document counts without network access."""

    DEFAULT_COUNTS = {
        "animal": 1_000_000,
        "mammal": 800_000,
        "dog": 500_000,
        "cat": 400_000,
        "animal mammal": 300_000,
        "animal dog": 200_000,
        "animal cat": 150_000,
        "mammal dog": 100_000,
        "mammal cat": 80_000,
        "dog cat": 50_000,
    }

    def __init__(
        self,
        counts: Mapping[str, int] | None = None,
        *,
        corpus_size: int = 10**12,
        default_count: int = 1000,
        default_co_occurrence_count: int = 0,
        mock_data: Mapping[str, int] | None = None,
    ) -> None:
        supplied = counts if counts is not None else mock_data
        self.counts = {
            self._normalize(query): max(0, int(count))
            for query, count in (supplied or {}).items()
        }
        self._corpus_size = max(1, int(corpus_size))
        self.default_count = max(0, int(default_count))
        self.default_co_occurrence_count = max(
            0, int(default_co_occurrence_count)
        )

    @staticmethod
    def _normalize(query: str) -> str:
        return (query or "").lower().strip().strip('"')

    def search_count(self, query: str) -> int:
        normalized = self._normalize(query)
        return self.counts.get(
            normalized,
            self.DEFAULT_COUNTS.get(normalized, self.default_count),
        )

    def co_occurrence_count(self, term1: str, term2: str) -> int:
        first, second = self._normalize(term1), self._normalize(term2)
        for pair in (f"{first} {second}", f"{second} {first}"):
            if pair in self.counts:
                return self.counts[pair]
            if pair in self.DEFAULT_COUNTS:
                return self.DEFAULT_COUNTS[pair]
        return min(
            self.default_co_occurrence_count,
            self.search_count(first),
            self.search_count(second),
        )

    def corpus_size(self) -> int:
        return self._corpus_size


@dataclass(frozen=True)
class VocabularyStats:
    total_terms: int
    unique_terms: int
    term_frequencies: dict[str, int]
    vocabulary_coverage: float


class DatasetSearchProvider(StaticSearchProvider):
    """Generate repeatable offline counts from a graph vocabulary."""

    def __init__(
        self,
        concept_graph: ConceptGraph,
        base_frequency: int = 1000,
        frequency_decay: float = 0.8,
        co_occurrence_probability: float = 0.3,
        *,
        seed: int | None = 0,
        corpus_size: int = 10**12,
    ) -> None:
        self.concept_graph = concept_graph
        self.base_frequency = max(1, int(base_frequency))
        self.frequency_decay = max(0.0, float(frequency_decay))
        self.co_occurrence_probability = min(
            1.0, max(0.0, float(co_occurrence_probability))
        )
        self.seed = seed
        self.vocabulary_stats = self._analyze_vocabulary()
        super().__init__(self._generate_counts(), corpus_size=corpus_size)

    def _analyze_vocabulary(self) -> VocabularyStats:
        frequencies: dict[str, int] = {}
        total = 0
        for node in self.concept_graph.nodes.values():
            for concept in getattr(node, "equiv_concepts", []):
                ground_set = getattr(concept, "ground_set", {}) or {}
                terms = [
                    *ground_set.get("labels", []),
                    *ground_set.get("exact_synonyms", []),
                    getattr(concept, "name", None),
                ]
                for candidate in terms:
                    if isinstance(candidate, str) and candidate.strip():
                        term = candidate.strip().lower()
                        frequencies[term] = frequencies.get(term, 0) + 1
                        total += 1
        return VocabularyStats(
            total_terms=total,
            unique_terms=len(frequencies),
            term_frequencies=frequencies,
            vocabulary_coverage=len(frequencies) / total if total else 0.0,
        )

    def _generate_counts(self) -> dict[str, int]:
        randomizer = random.Random(self.seed)
        terms = sorted(
            self.vocabulary_stats.term_frequencies,
            key=lambda term: (
                -self.vocabulary_stats.term_frequencies[term],
                term,
            ),
        )
        counts = {
            term: max(1, int(self.base_frequency * self.frequency_decay**index))
            for index, term in enumerate(terms)
        }
        for index, first in enumerate(terms):
            for second in terms[index + 1 :]:
                if randomizer.random() < self.co_occurrence_probability:
                    counts[f"{first} {second}"] = max(
                        1,
                        int(
                            randomizer.uniform(0.1, 0.5)
                            * min(counts[first], counts[second])
                        ),
                    )
        return counts

    def get_vocabulary_stats(self) -> VocabularyStats:
        return self.vocabulary_stats
