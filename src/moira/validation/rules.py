"""Composable validation rules for ontology alignments."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, TypeAlias

Match: TypeAlias = tuple[str, str]


class ValidationSeverity(str, Enum):
    """Severity assigned to a validation result."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class RuleResult:
    """Result of applying one rule."""

    rule_name: str
    severity: ValidationSeverity
    passed: bool
    message: str
    matches_checked: int
    duplicate_count: int = 0


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Immutable aggregate of validation rule results."""

    results: tuple[RuleResult, ...]

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)


class ValidationRule(Protocol):
    """Interface implemented by alignment validation rules."""

    name: str
    severity: ValidationSeverity

    def validate(self, matches: Sequence[Match]) -> RuleResult: ...


@dataclass(frozen=True, slots=True)
class DuplicateMatchRule:
    """Reject repeated alignment pairs in any input iterable."""

    name: str = "No Duplicate Matches"
    severity: ValidationSeverity = ValidationSeverity.ERROR

    def validate(self, matches: Sequence[Match]) -> RuleResult:
        duplicate_count = sum(count - 1 for count in Counter(matches).values())
        return RuleResult(
            rule_name=self.name,
            severity=self.severity,
            passed=duplicate_count == 0,
            message=(
                f"Found {duplicate_count} duplicate matches"
                if duplicate_count
                else "No duplicates found"
            ),
            matches_checked=len(matches),
            duplicate_count=duplicate_count,
        )


# Basic rule instances for immediate use
BASIC_RULES: tuple[ValidationRule, ...] = (DuplicateMatchRule(),)


def validate_matches(
    matches: Iterable[Match],
    rules: Iterable[ValidationRule] = BASIC_RULES,
) -> ValidationReport:
    """Materialize an iterable once, then apply all supplied rules."""
    materialized = tuple(matches)
    return ValidationReport(
        results=tuple(rule.validate(materialized) for rule in rules)
    )