"""Concept paths stored as SPARQL patterns for the query cache."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

SUBCLASS_OF = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
RELATED = "http://agentoi.org/related"


def normalize_question(text: str) -> str:
    """Fold a question into the hot-cache key."""
    collapsed = re.sub(r"\s+", " ", text.strip().lower())
    return collapsed.strip(" ?.!")


def _term(value: str) -> str:
    if value.startswith(("http://", "https://")):
        iri = value.replace(">", "%3E")
        return f"<{iri}>"
    safe = quote(value.replace(" ", "_"), safe="")
    return f"<http://agentoi.org/concept/{safe}>"


def render_sparql(
    concept_ids: Sequence[str],
    edges: Sequence[tuple[str, str, str]] = (),
) -> str:
    """Render the concept path as a SPARQL basic graph pattern."""
    lines: list[str] = []
    index = {concept_id: position for position, concept_id in enumerate(concept_ids)}
    for position, concept_id in enumerate(concept_ids):
        lines.append(f"?c{position} a {_term(concept_id)} .")
    for source, predicate, target in edges:
        if source not in index or target not in index:
            continue
        lines.append(f"?c{index[source]} <{predicate}> ?c{index[target]} .")
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class QueryPath:
    """One retrieved concept sequence and its SPARQL pattern."""

    question: str
    concept_ids: tuple[str, ...]
    edges: tuple[tuple[str, str, str], ...]
    sparql: str

    @classmethod
    def from_steps(
        cls,
        question: str,
        concept_ids: Sequence[str],
        edges: Sequence[tuple[str, str, str]] = (),
    ) -> QueryPath:
        concepts = tuple(concept_ids)
        kept = tuple(
            (source, predicate, target)
            for source, predicate, target in edges
            if source in concepts and target in concepts
        )
        return cls(
            question=normalize_question(question),
            concept_ids=concepts,
            edges=kept,
            sparql=render_sparql(concepts, kept),
        )


def query_path_from_graph(
    question: str,
    concept_ids: Sequence[str],
    graph: Any,
) -> QueryPath:
    """Build a path from the induced edges among the selected concepts."""
    selected = set(concept_ids)
    edges: list[tuple[str, str, str]] = []
    for edge in getattr(graph, "edges", ()):
        source = edge.src.id
        target = edge.tgt.id
        if source not in selected or target not in selected:
            continue
        predicate = SUBCLASS_OF if float(edge.score) >= 2 else RELATED
        edges.append((source, predicate, target))
    return QueryPath.from_steps(question, concept_ids, edges)
