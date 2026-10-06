"""Normalization and deterministic evaluation of ontology tuples."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import TypeAlias

OntologyTuple: TypeAlias = tuple[str, str, str]
TupleCondition: TypeAlias = Callable[[OntologyTuple], bool]
TupleTransform: TypeAlias = Callable[[OntologyTuple], OntologyTuple]


def _as_ontology_tuple(value: Sequence[str]) -> OntologyTuple:
    if len(value) != 3:
        raise ValueError("An ontology tuple must contain exactly three values")
    if not all(isinstance(item, str) for item in value):
        raise TypeError("Ontology tuple values must be strings")
    return value[0], value[1], value[2]


@dataclass(frozen=True, slots=True)
class NormalizationRule:
    """One named tuple transformation."""

    name: str
    condition: TupleCondition
    transform: TupleTransform
    priority: int = 0

    def apply(self, value: OntologyTuple) -> OntologyTuple:
        return _as_ontology_tuple(self.transform(value))


def _normalize_relation(value: OntologyTuple) -> OntologyTuple:
    source, relation, target = value
    aliases = {
        "is_a": "isA",
        "is-a": "isA",
        "type": "isA",
        "sub_class_of": "subClassOf",
        "sub-class-of": "subClassOf",
        "same_as": "sameAs",
        "same-as": "sameAs",
        "identical": "sameAs",
        "has_property": "hasProperty",
        "has-property": "hasProperty",
    }
    return source, aliases.get(relation.casefold(), relation), target


def _normalize_concepts(value: OntologyTuple) -> OntologyTuple:
    source, relation, target = value

    def normalize(value: str) -> str:
        return " ".join(word.capitalize() for word in value.replace("_", " ").split())

    return normalize(source), relation, normalize(target)


DEFAULT_NORMALIZATION_RULES = (
    NormalizationRule("normalize_relation", lambda _: True, _normalize_relation, 20),
    NormalizationRule("normalize_concepts", lambda _: True, _normalize_concepts, 10),
)


class TupleNormalizer:
    """Apply ordered normalization rules with stable tuple cache keys."""

    def __init__(
        self, rules: Iterable[NormalizationRule] = DEFAULT_NORMALIZATION_RULES
    ) -> None:
        self._rules = sorted(rules, key=lambda rule: rule.priority, reverse=True)
        self._cache: dict[OntologyTuple, OntologyTuple] = {}

    def add_rule(self, rule: NormalizationRule) -> None:
        self._rules.append(rule)
        self._rules.sort(key=lambda item: item.priority, reverse=True)
        self._cache.clear()

    def normalize(self, value: Sequence[str]) -> OntologyTuple:
        key = _as_ontology_tuple(value)
        if key not in self._cache:
            normalized = key
            for rule in self._rules:
                if rule.condition(normalized):
                    normalized = rule.apply(normalized)
            self._cache[key] = normalized
        return self._cache[key]

    def normalize_all(self, values: Iterable[Sequence[str]]) -> tuple[OntologyTuple, ...]:
        return tuple(self.normalize(value) for value in values)


@dataclass(frozen=True, slots=True)
class TupleEvaluation:
    """Immutable component scores for one ontology tuple."""

    value: OntologyTuple
    confidence: float
    semantic_score: float
    graph_score: float
    overall_score: float


@dataclass(frozen=True, slots=True)
class EvaluationStatistics:
    """Summary of a tuple evaluation batch."""

    count: int
    mean_score: float
    score_stddev: float
    mean_confidence: float
    high_confidence_ratio: float
    high_score_ratio: float


class TupleEvaluator:
    """Score ontology tuples using lexical and relation consistency."""

    _RELATION_WEIGHTS = {
        "isa": 0.9,
        "subclassof": 0.9,
        "instanceof": 0.8,
        "hasproperty": 0.7,
        "relatedto": 0.6,
        "sameas": 0.95,
        "equivalentto": 0.95,
        "similarto": 0.8,
    }

    def __init__(self) -> None:
        self._cache: dict[OntologyTuple, TupleEvaluation] = {}

    @staticmethod
    def semantic_similarity(first: str, second: str) -> float:
        left = first.casefold()
        right = second.casefold()
        if left == right:
            return 1.0

        left_words = set(left.split())
        right_words = set(right.split())
        if not left_words or not right_words:
            return 0.0

        overlap = len(left_words & right_words) / len(left_words | right_words)
        substring_bonus = 0.3 if left in right or right in left else 0.0
        return min(1.0, overlap + substring_bonus)

    def _graph_score(self, value: OntologyTuple) -> float:
        source, relation, target = value
        relation_key = relation.casefold()
        relation_score = self._RELATION_WEIGHTS.get(relation_key, 0.5)
        similarity = self.semantic_similarity(source, target)
        if relation_key in {"sameas", "equivalentto"} and similarity > 0.8:
            return min(1.0, relation_score + 0.1)
        if relation_key in {"isa", "subclassof"} and similarity > 0.5:
            return relation_score
        return relation_score * 0.8

    def evaluate(self, value: Sequence[str]) -> TupleEvaluation:
        key = _as_ontology_tuple(value)
        if key not in self._cache:
            source, _, target = key
            semantic = self.semantic_similarity(source, target)
            graph = self._graph_score(key)
            consistency = max(0.0, 1.0 - abs(semantic - graph))
            name_length = min(1.0, (len(source) + len(target)) / 20)
            confidence = min(1.0, ((semantic + graph) / 2 + consistency + name_length) / 3)
            base_score = 0.6 * semantic + 0.4 * graph
            overall = min(1.0, base_score * (0.5 + 0.5 * confidence))
            self._cache[key] = TupleEvaluation(
                value=key,
                confidence=confidence,
                semantic_score=semantic,
                graph_score=graph,
                overall_score=overall,
            )
        return self._cache[key]

    def evaluate_all(
        self, values: Iterable[Sequence[str]]
    ) -> tuple[TupleEvaluation, ...]:
        return tuple(self.evaluate(value) for value in values)

    @staticmethod
    def statistics(results: Iterable[TupleEvaluation]) -> EvaluationStatistics:
        items = tuple(results)
        if not items:
            return EvaluationStatistics(0, 0.0, 0.0, 0.0, 0.0, 0.0)
        scores = tuple(item.overall_score for item in items)
        confidences = tuple(item.confidence for item in items)
        return EvaluationStatistics(
            count=len(items),
            mean_score=fmean(scores),
            score_stddev=pstdev(scores),
            mean_confidence=fmean(confidences),
            high_confidence_ratio=sum(value > 0.7 for value in confidences) / len(items),
            high_score_ratio=sum(value > 0.8 for value in scores) / len(items),
        )
