"""Normalized Google Distance (NGD) similarity."""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

import torch

from agentoi.retrieval.web import StaticSearchProvider, WebSearchProvider
from agentoi.retrieval.web.base import LRUCache

if TYPE_CHECKING:
    from agentoi.algorithms.graph import ConceptGraph, EquivalentClass


class NormalizedGoogleDistance:
    """Compute NGD similarity with an explicit search-count provider."""

    def __init__(
        self,
        search_provider: WebSearchProvider | None = None,
        cache_size: int = 1000,
        min_count: int = 1,
    ):
        self.search_provider = search_provider or StaticSearchProvider()
        self.min_count = max(1, int(min_count))
        self._search_cache = LRUCache(cache_size)
        self._ngd_cache = LRUCache(cache_size)

    @staticmethod
    def _normalize_term(term: str) -> str:
        return (term or "").split("/")[-1].replace("_", " ")

    def _search_count_cached(self, query: str) -> int:
        cached = self._search_cache.get_cached(("term", query))
        if cached is not None:
            return cached
        try:
            count = max(0, int(self.search_provider.search_count(query)))
        except Exception:
            count = 0
        if count > 0:
            self._search_cache.put_cached(("term", query), count)
        return count

    def _co_count_cached(self, term1: str, term2: str) -> int:
        key = ("pair", *sorted((term1, term2)))
        cached = self._search_cache.get_cached(key)
        if cached is not None:
            return cached
        try:
            count = max(
                0, int(self.search_provider.co_occurrence_count(term1, term2))
            )
        except Exception:
            count = 0
        if count > 0:
            self._search_cache.put_cached(key, count)
        return count

    def _corpus_size(self) -> int:
        try:
            corpus_size = getattr(self.search_provider, "corpus_size")
            value = corpus_size() if callable(corpus_size) else corpus_size
            return max(1, int(value))
        except Exception:
            return 1

    def compute_ngd(self, term1: str, term2: str) -> float:
        term1 = self._normalize_term(term1)
        term2 = self._normalize_term(term2)
        if term1 and term1.casefold() == term2.casefold():
            return 0.0
        key = tuple(sorted((term1, term2)))
        cached = self._ngd_cache.get_cached(key)
        if cached is not None:
            return cached

        frequency1 = self._search_count_cached(term1)
        frequency2 = self._search_count_cached(term2)
        co_frequency = self._co_count_cached(term1, term2)
        corpus_size = self._corpus_size()

        if min(frequency1, frequency2, co_frequency) < self.min_count:
            ngd = 1.0
        else:
            denominator = math.log(corpus_size) - math.log(
                min(frequency1, frequency2)
            )
            if denominator <= 0:
                ngd = 1.0
            else:
                numerator = math.log(max(frequency1, frequency2)) - math.log(
                    co_frequency
                )
                ngd = numerator / denominator

        ngd = max(0.0, min(1.0, float(ngd)))
        if min(frequency1, frequency2, co_frequency) >= self.min_count:
            self._ngd_cache.put_cached(key, ngd)
        return ngd

    def compute_similarity(self, term1: str, term2: str) -> float:
        return 1.0 - self.compute_ngd(term1, term2)

    def _extract_terms_from_node(self, node: EquivalentClass) -> list[str]:
        terms: list[str] = []
        for concept in getattr(node, "equiv_concepts", []):
            ground_set = getattr(concept, "ground_set", {}) or {}
            terms.extend(
                term
                for term in ground_set.get("labels", [])
                if isinstance(term, str)
            )
            terms.extend(
                term
                for term in ground_set.get("exact_synonyms", [])
                if isinstance(term, str)
            )
            name = getattr(concept, "name", None)
            if isinstance(name, str):
                terms.append(name)

        seen: set[str] = set()
        unique_terms = []
        for term in terms:
            stripped = term.strip()
            normalized = stripped.lower()
            if stripped and normalized not in seen:
                seen.add(normalized)
                unique_terms.append(stripped)
        return unique_terms

    @staticmethod
    def _score_term_for_query(term: str) -> tuple[int, int]:
        identifier_pattern = re.compile(
            r"https?://|^obo:|^[A-Z_]+\d+$|[_/]|^\w+:\w+$"
        )
        stripped = term.strip()
        return (1 if identifier_pattern.search(stripped) else 0, len(stripped))

    def representative_term(self, node: EquivalentClass) -> str:
        """Return the most human-readable term available for a node."""
        terms = self._extract_terms_from_node(node)
        if not terms:
            return getattr(node, "id", "")
        return min(terms, key=self._score_term_for_query)

    @staticmethod
    def _valid_node_ids(
        graph: ConceptGraph, node_ids: list[str] | None
    ) -> list[str]:
        requested = list(graph.nodes) if node_ids is None else list(node_ids)
        return [node_id for node_id in requested if node_id in graph.nodes]

    def compute_pairwise_similarity(
        self,
        graph: ConceptGraph,
        node_ids: list[str] | None = None,
    ) -> torch.Tensor:
        valid_ids = self._valid_node_ids(graph, node_ids)
        matrix = torch.zeros((len(valid_ids), len(valid_ids)))
        terms = [
            self.representative_term(graph.nodes[node_id]) for node_id in valid_ids
        ]
        for row, term1 in enumerate(terms):
            for column in range(row, len(terms)):
                similarity = self.compute_similarity(term1, terms[column])
                matrix[row, column] = similarity
                matrix[column, row] = similarity
        return matrix

    def compute_similarity_to_query(
        self,
        graph: ConceptGraph,
        query_term: str,
        node_ids: list[str] | None = None,
    ) -> torch.Tensor:
        valid_ids = self._valid_node_ids(graph, node_ids)
        similarities = torch.zeros(len(valid_ids))
        for index, node_id in enumerate(valid_ids):
            similarities[index] = self.compute_similarity(
                query_term, self.representative_term(graph.nodes[node_id])
            )
        return similarities

    def get_top_similar_pairs(
        self,
        graph: ConceptGraph,
        k: int = 10,
        node_ids: list[str] | None = None,
        exclude_self: bool = True,
    ) -> list[tuple[str, str, float]]:
        valid_ids = self._valid_node_ids(graph, node_ids)
        if k <= 0 or not valid_ids:
            return []
        matrix = self.compute_pairwise_similarity(graph, valid_ids)
        diagonal = 1 if exclude_self else 0
        coordinates = torch.nonzero(
            torch.triu(torch.ones_like(matrix), diagonal=diagonal).bool(),
            as_tuple=False,
        )
        if not len(coordinates):
            return []
        values = matrix[coordinates[:, 0], coordinates[:, 1]]
        top_values, selected = torch.topk(values, k=min(k, len(values)))
        return [
            (
                valid_ids[coordinates[position, 0].item()],
                valid_ids[coordinates[position, 1].item()],
                top_values[rank].item(),
            )
            for rank, position in enumerate(selected)
        ]

    def get_most_similar_nodes(
        self,
        graph: ConceptGraph,
        query_term: str,
        k: int = 10,
        node_ids: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        valid_ids = self._valid_node_ids(graph, node_ids)
        if k <= 0 or not valid_ids:
            return []
        similarities = self.compute_similarity_to_query(
            graph, query_term, valid_ids
        )
        top_values, top_indices = torch.topk(
            similarities, k=min(k, len(similarities))
        )
        return [
            (valid_ids[index.item()], top_values[rank].item())
            for rank, index in enumerate(top_indices)
        ]


def compute_google_similarity_matrix(
    graph: ConceptGraph,
    node_ids: list[str] | None = None,
    search_provider: WebSearchProvider | None = None,
) -> torch.Tensor:
    return NormalizedGoogleDistance(search_provider).compute_pairwise_similarity(
        graph, node_ids
    )


def compute_google_similarity_to_query(
    graph: ConceptGraph,
    query_term: str,
    node_ids: list[str] | None = None,
    search_provider: WebSearchProvider | None = None,
) -> torch.Tensor:
    return NormalizedGoogleDistance(
        search_provider
    ).compute_similarity_to_query(graph, query_term, node_ids)
