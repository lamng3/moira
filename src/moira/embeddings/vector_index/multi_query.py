"""Shared multi-query search used by graph-like data structures."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from .base import BaseVectorIndex
from .registry import create_vector_index


class MultiQuerySearchService:
    """Search graph nodes through a local or prebuilt vector index."""

    def __init__(
        self,
        index: BaseVectorIndex | None = None,
        *,
        backend: str = "torch-exact",
        backend_options: dict[str, Any] | None = None,
    ):
        self.index = index
        self.backend = backend
        self.backend_options = dict(backend_options or {})

    @staticmethod
    def index_nodes(
        index: BaseVectorIndex,
        nodes: Sequence[Any],
        *,
        clear: bool = False,
    ) -> BaseVectorIndex:
        """Upsert graph nodes into an index using stable node identifiers."""
        if clear:
            index.clear()
        if not nodes:
            return index
        vectors = torch.stack(
            [torch.as_tensor(node.embedding) for node in nodes]
        )
        index.add(
            [node.id for node in nodes],
            vectors,
            [{"position": position} for position, _ in enumerate(nodes)],
        )
        return index

    @staticmethod
    def _matches_prefix(
        node: Any,
        prefixes: tuple[str, ...],
        exclusive: bool,
    ) -> bool:
        concepts = getattr(node, "equiv_concepts", ())

        def matches(concept: Any) -> bool:
            iri = (getattr(concept, "iri", None) or "").strip()
            name = (getattr(concept, "name", None) or "").strip()
            return any(
                iri.startswith(prefix) or name.startswith(prefix)
                for prefix in prefixes
            )

        if exclusive:
            return bool(concepts) and all(matches(concept) for concept in concepts)
        return any(matches(concept) for concept in concepts)

    def search(
        self,
        nodes: Sequence[Any],
        terms: str | Sequence[str],
        *,
        text_model: Any,
        k: int = 10,
        agg: str = "max",
        weights: Sequence[float] | None = None,
        return_out: bool = True,
        iri_prefix: str | Sequence[str] | None = None,
        exclusive: bool = False,
    ) -> list[tuple[Any, float]] | tuple[torch.Tensor, torch.Tensor]:
        if not nodes:
            return []

        if isinstance(terms, str):
            queries = [terms.strip()] if terms.strip() else []
        else:
            queries = [str(term).strip() for term in terms if str(term).strip()]
        if not queries:
            return []

        if iri_prefix is None:
            prefixes: tuple[str, ...] = ()
        elif isinstance(iri_prefix, str):
            prefixes = (iri_prefix,)
        else:
            prefixes = tuple(iri_prefix)

        candidates = list(nodes)
        if prefixes:
            candidates = [
                node
                for node in candidates
                if self._matches_prefix(node, prefixes, exclusive)
            ]
        if not candidates:
            return []

        queries_as_vectors = [
            torch.as_tensor(text_model.to_embedding(query)) for query in queries
        ]

        if self.index is None:
            for node in candidates:
                if getattr(node, "embedding", None) is None:
                    raise RuntimeError(
                        "Call compute_all_embeddings() before nearest_nodes()."
                    )
            vectors = torch.stack(
                [torch.as_tensor(node.embedding) for node in candidates]
            )
            options = {
                **self.backend_options,
                "dimension": int(vectors.shape[1]),
            }
            index = self.index_nodes(
                create_vector_index(self.backend, **options), candidates
            )
        else:
            index = self.index
        allowed_ids = {node.id for node in candidates}
        results = index.search_many(
            queries_as_vectors,
            k=min(k, len(candidates)),
            agg=agg,
            weights=weights,
            allowed_ids=allowed_ids,
        )
        node_by_id = {node.id: node for node in candidates}
        position_by_id = {
            node.id: position for position, node in enumerate(candidates)
        }
        results = [result for result in results if result.id in node_by_id][:k]

        if return_out:
            return [
                (node_by_id[result.id], result.score)
                for result in results
            ]

        reference_embedding = getattr(candidates[0], "embedding", None)
        result_dtype = (
            torch.as_tensor(reference_embedding).dtype
            if reference_embedding is not None
            else torch.float32
        )
        values = torch.tensor(
            [result.score for result in results],
            dtype=result_dtype,
        )
        positions = torch.tensor(
            [position_by_id[result.id] for result in results],
            dtype=torch.long,
        )
        return values, positions
