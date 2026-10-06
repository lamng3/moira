"""Uncertainty scoring with dependency-free defaults and lazy optional backends."""

from __future__ import annotations

from typing import Any

from moira.uq.agreement import AgreementConfidenceScorer
from moira.uq.common import ResponseScorer, ScoreOutput, parse_yes_no

__all__ = [
    "AgreementConfidenceScorer",
    "ResponseScorer",
    "ScoreOutput",
    "parse_yes_no",
    "UQLMConfidenceScorer",
    "GraphConfidenceScorer",
    "EnsembleConfidenceScorer",
    "SCORER_EXACT",
    "SCORER_MULTI",
    "SCORER_FULL",
    "GRAPH_SCORERS",
    "DEFAULT_WEIGHTS",
]


def __getattr__(name: str) -> Any:
    if name in {"UQLMConfidenceScorer", "SCORER_EXACT", "SCORER_MULTI", "SCORER_FULL"}:
        from moira.uq import scorer

        return getattr(scorer, name)
    if name in {"GraphConfidenceScorer", "GRAPH_SCORERS"}:
        from moira.uq import graph_scorer

        return getattr(graph_scorer, name)
    if name in {"EnsembleConfidenceScorer", "DEFAULT_WEIGHTS"}:
        from moira.uq import ensemble_scorer

        return getattr(ensemble_scorer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
