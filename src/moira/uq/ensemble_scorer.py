"""Composable uncertainty ensemble using only public scorer methods."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from moira.uq.agreement import AgreementConfidenceScorer
from moira.uq.common import (
    ResponseScorer,
    ScoreOutput,
    clamp_score,
    generate_responses,
    majority_result,
)

DEFAULT_WEIGHTS: dict[str, float] = {"exact_match": 1.0}


class EnsembleConfidenceScorer:
    """Combine score dictionaries from injected response scorers."""

    def __init__(
        self,
        llm: Any | None = None,
        num_responses: int = 5,
        weights: Mapping[str, float] | None = None,
        scorers: Sequence[ResponseScorer] | None = None,
    ) -> None:
        if isinstance(num_responses, bool) or not isinstance(num_responses, int):
            raise TypeError("num_responses must be an integer")
        if num_responses < 1:
            raise ValueError("num_responses must be positive")
        selected = tuple(
            (AgreementConfidenceScorer(),) if scorers is None else scorers
        )
        if not selected:
            raise ValueError("at least one scorer is required")
        if any(not isinstance(scorer, ResponseScorer) for scorer in selected):
            raise TypeError(
                "each scorer must implement score_prompt_from_responses(responses)"
            )
        self._weights = _validate_weights(
            DEFAULT_WEIGHTS if weights is None else weights
        )
        self._scorers = selected
        self.llm = llm
        self.num_responses = num_responses

    def score_prompt(self, prompt: str) -> ScoreOutput:
        return self.score_prompt_from_responses(
            generate_responses(self.llm, prompt, self.num_responses)
        )

    def score_prompt_from_responses(
        self, raw_responses: Sequence[str]
    ) -> ScoreOutput:
        answer, _ = majority_result(raw_responses)
        scores: dict[str, float] = {}
        for scorer in self._scorers:
            _, _, component_scores = scorer.score_prompt_from_responses(raw_responses)
            for name, value in component_scores.items():
                if not isinstance(name, str) or not name:
                    raise ValueError("scorer names must be non-empty strings")
                if name in scores:
                    raise ValueError(f"duplicate scorer name: {name}")
                scores[name] = clamp_score(value, name=name)

        available = {
            name: weight for name, weight in self._weights.items() if name in scores
        }
        total_weight = sum(available.values())
        if total_weight <= 0:
            names = ", ".join(sorted(scores)) or "(none)"
            raise ValueError(f"no positive weight matches available scores: {names}")
        confidence = clamp_score(
            sum(scores[name] * weight for name, weight in available.items())
            / total_weight,
            name="ensemble",
        )
        scores["ensemble"] = confidence
        return answer, confidence, scores


def _validate_weights(weights: Mapping[str, float]) -> dict[str, float]:
    if not weights:
        raise ValueError("weights must not be empty")
    result: dict[str, float] = {}
    for name, weight in weights.items():
        if not isinstance(name, str) or not name:
            raise ValueError("weight names must be non-empty strings")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise TypeError(f"weight for {name} must be a number")
        value = float(weight)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"weight for {name} must be finite and non-negative")
        result[name] = value
    if not any(result.values()):
        raise ValueError("at least one weight must be positive")
    return result
