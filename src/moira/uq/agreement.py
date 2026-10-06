"""Dependency-free agreement scoring for binary responses."""

from __future__ import annotations

from typing import Any, Sequence

from moira.uq.common import ScoreOutput, generate_responses, majority_result


class AgreementConfidenceScorer:
    """Score confidence as the winning YES/NO response fraction."""

    def __init__(self, llm: Any | None = None, num_responses: int = 5) -> None:
        if isinstance(num_responses, bool) or not isinstance(num_responses, int):
            raise TypeError("num_responses must be an integer")
        if num_responses < 1:
            raise ValueError("num_responses must be positive")
        self.llm = llm
        self.num_responses = num_responses

    def score_prompt(self, prompt: str) -> ScoreOutput:
        return self.score_prompt_from_responses(
            generate_responses(self.llm, prompt, self.num_responses)
        )

    def score_prompt_from_responses(
        self, raw_responses: Sequence[str]
    ) -> ScoreOutput:
        answer, agreement = majority_result(raw_responses)
        return answer, agreement, {"exact_match": agreement}
