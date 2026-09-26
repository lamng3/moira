"""Metrics for evaluating predicted ontology alignments."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeAlias

Alignment: TypeAlias = tuple[str, str]


def calculate_precision(true_positives: int, false_positives: int) -> float:
    """Return ``TP / (TP + FP)``."""
    total_predicted = true_positives + false_positives
    if total_predicted == 0:
        return 0.0
    return true_positives / total_predicted


def calculate_recall(true_positives: int, false_negatives: int) -> float:
    """Return ``TP / (TP + FN)``."""
    total_actual = true_positives + false_negatives
    if total_actual == 0:
        return 0.0
    return true_positives / total_actual


def calculate_f1_score(precision: float, recall: float) -> float:
    """Return the harmonic mean of precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


@dataclass(frozen=True, slots=True)
class AlignmentMetrics:
    """Immutable confusion counts and derived alignment metrics."""

    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        return calculate_precision(self.true_positives, self.false_positives)

    @property
    def recall(self) -> float:
        return calculate_recall(self.true_positives, self.false_negatives)

    @property
    def f1_score(self) -> float:
        return calculate_f1_score(self.precision, self.recall)

    @property
    def total_predictions(self) -> int:
        return self.true_positives + self.false_positives

    @property
    def total_gold(self) -> int:
        return self.true_positives + self.false_negatives


def evaluate_alignment(
    predicted: Iterable[Alignment],
    gold: Iterable[Alignment],
) -> AlignmentMetrics:
    """Compare two alignment iterables using set membership semantics."""
    predicted_set = frozenset(predicted)
    gold_set = frozenset(gold)
    return AlignmentMetrics(
        true_positives=len(predicted_set & gold_set),
        false_positives=len(predicted_set - gold_set),
        false_negatives=len(gold_set - predicted_set),
    )
