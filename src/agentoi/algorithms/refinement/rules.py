"""Rule loading and relation normalization for ontology refinement."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

from agentoi.algorithms.graph.predicates import canonicalize_predicate

Triple: TypeAlias = tuple[str, str, str]  # noqa: UP040


def _predicate_token(value: object) -> str:
    return canonicalize_predicate(str(value or ""))


@dataclass
class RuleSet:
    """Normalized rules used by :class:`RelationRefiner`."""

    normalize: dict[str, str] = field(default_factory=dict)
    inverse: dict[str, str] = field(default_factory=dict)
    transitive: set[str] = field(default_factory=set)
    drop: set[str] = field(default_factory=set)
    grammar: dict[str, str] = field(default_factory=dict)
    domain_range: dict[str, tuple[set[str], set[str]]] = field(default_factory=dict)
    grammar_flip: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.normalize = {
            _predicate_token(key): _predicate_token(value)
            for key, value in self.normalize.items()
            if _predicate_token(key) and _predicate_token(value)
        }
        self.grammar = {
            _predicate_token(key): _predicate_token(value)
            for key, value in self.grammar.items()
            if _predicate_token(key) and _predicate_token(value)
        }
        self.inverse = {
            self.canonical_predicate(key): self.canonical_predicate(value)
            for key, value in self.inverse.items()
            if _predicate_token(key) and _predicate_token(value)
        }
        self.transitive = {
            self.inverse.get(canonical, canonical)
            for value in self.transitive
            if (canonical := self.canonical_predicate(value))
        }
        self.drop = {self.canonical_predicate(value) for value in self.drop}
        self.grammar_flip = {
            self.canonical_predicate(value) for value in self.grammar_flip
        }
        self.domain_range = {
            self.canonical_predicate(predicate): (set(domain), set(range_))
            for predicate, (domain, range_) in self.domain_range.items()
        }

    def canonical_predicate(self, predicate: object) -> str:
        token = _predicate_token(predicate)
        token = self.normalize.get(token, token)
        return self.grammar.get(token, token)

    # Backward-compatible spelling used by the original procedural module.
    canon_pred = canonical_predicate

    def normalize_triple(self, triple: Triple) -> Triple | None:
        """Clean, canonicalize, orient, and optionally reject one triple."""
        subject, predicate, object_ = (str(value or "").strip() for value in triple)
        predicate = self.canonical_predicate(predicate)
        if not subject or not predicate or not object_ or predicate in self.drop:
            return None
        if predicate in self.inverse:
            subject, object_ = object_, subject
            predicate = self.inverse[predicate]
        if predicate in self.grammar_flip or _looks_passive(predicate):
            subject, object_ = object_, subject
        return subject, predicate, object_

    def normalize_triples(self, triples: Iterable[Triple]) -> list[Triple]:
        """Normalize and deduplicate triples while preserving input order."""
        result: list[Triple] = []
        seen: set[Triple] = set()
        for raw in triples:
            triple = self.normalize_triple(raw)
            if triple is not None and triple not in seen:
                seen.add(triple)
                result.append(triple)
        return result


def _looks_passive(predicate: str) -> bool:
    return predicate.endswith("_by") and predicate.startswith(("is_", "was_"))


def default_rules() -> RuleSet:
    return RuleSet(
        normalize={
            "subclassof": "subclassof",
            "rdfs:subclassof": "subclassof",
            "broader": "subclassof",
            "is_a": "subclassof",
            "isa": "subclassof",
            "is-a": "subclassof",
            "narrower": "superclassof",
            "partof": "part_of",
            "part_of": "part_of",
            "obo:part_of": "part_of",
            "has_part": "has_part",
            "equivalentclass": "equivalentclass",
            "owl:equivalentclass": "equivalentclass",
            "sameas": "sameas",
            "owl:sameas": "sameas",
            "exactmatch": "exactmatch",
            "skos:exactmatch": "exactmatch",
            "equivalentto": "equivalentto",
        },
        inverse={"superclassof": "subclassof", "has_part": "part_of"},
        transitive={"subclassof", "part_of"},
    )


def _domain_range(value: object) -> tuple[set[str], set[str]]:
    if isinstance(value, Mapping):
        return set(value.get("domain", [])), set(value.get("range", []))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return set(value[0]), set(value[1])
    raise ValueError("domain_range entries must contain domain and range collections")


def merge_rules(base: RuleSet, data: Mapping[str, Any]) -> RuleSet:
    """Merge every supported JSON rule field into ``base``."""
    domain_range = dict(base.domain_range)
    domain_range.update(
        {
            str(predicate): _domain_range(value)
            for predicate, value in data.get("domain_range", {}).items()
        }
    )
    return RuleSet(
        normalize={**base.normalize, **data.get("normalize", {})},
        inverse={**base.inverse, **data.get("inverse", {})},
        transitive=base.transitive | set(data.get("transitive", [])),
        drop=base.drop | set(data.get("drop", [])),
        grammar={**base.grammar, **data.get("grammar", {})},
        domain_range=domain_range,
        grammar_flip=base.grammar_flip | set(data.get("grammar_flip", [])),
    )


def load_rules(path: str | Path | None = None) -> RuleSet:
    """Load and merge a JSON rule file; no path means built-in defaults."""
    base = default_rules()
    if path is None:
        return base
    rule_path = Path(path)
    if not rule_path.is_file():
        return base
    with rule_path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, Mapping):
        raise TypeError("rule JSON must be an object")
    return merge_rules(base, data)


load_rules_from_json = load_rules
