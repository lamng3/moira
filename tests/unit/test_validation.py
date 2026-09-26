from dataclasses import FrozenInstanceError

import pytest

from agentoi.validation import (
    AlignmentMetrics,
    ValidationSeverity,
    evaluate_alignment,
    validate_matches,
)


def test_alignment_metrics_from_iterables() -> None:
    predicted = (match for match in (("A", "X"), ("B", "Y"), ("C", "W")))
    gold = [("A", "X"), ("B", "Y"), ("D", "Z")]

    metrics = evaluate_alignment(predicted, gold)

    assert metrics == AlignmentMetrics(2, 1, 1)
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(2 / 3)
    assert metrics.f1_score == pytest.approx(2 / 3)


def test_empty_alignment_has_zero_scores() -> None:
    metrics = evaluate_alignment([], [])

    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1_score == 0.0


def test_alignment_metrics_are_immutable() -> None:
    metrics = evaluate_alignment([("A", "X")], [("A", "X")])

    with pytest.raises(FrozenInstanceError):
        metrics.true_positives = 0


def test_duplicate_validation_accepts_generator_and_finds_repeats() -> None:
    matches = (match for match in (("A", "X"), ("B", "Y"), ("A", "X")))

    report = validate_matches(matches)

    assert not report.passed
    assert report.results[0].severity is ValidationSeverity.ERROR
    assert report.results[0].duplicate_count == 1
    assert report.results[0].matches_checked == 3


def test_duplicate_validation_accepts_clean_set() -> None:
    report = validate_matches({("A", "X"), ("B", "Y")})

    assert report.passed
    assert report.results[0].duplicate_count == 0
