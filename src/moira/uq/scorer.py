"""Optional UQLM black-box uncertainty backend."""

from __future__ import annotations

from typing import Any, Sequence

from moira.uq.common import (
    ScoreOutput,
    clamp_score,
    generate_responses,
    majority_result,
    parse_yes_no,
)

SCORER_EXACT = ["exact_match"]
SCORER_MULTI = [
    "exact_match",
    "noncontradiction",
    "semantic_negentropy",
    "cosine_sim",
]
SCORER_FULL = [
    "exact_match",
    "noncontradiction",
    "entailment",
    "semantic_negentropy",
    "semantic_sets_confidence",
    "bert_score",
    "cosine_sim",
]

class UQLMConfidenceScorer:
    """Compute binary confidence through the optional ``uqlm`` package."""

    SCORER_DEFAULTS = ["exact_match"]

    def __init__(
        self,
        llm: Any,
        num_responses: int = 5,
        scorers: Sequence[str] | None = None,
        primary_scorer: str = "exact_match",
        device: str | None = None,
        *,
        backend: Any | None = None,
    ) -> None:
        if isinstance(num_responses, bool) or not isinstance(num_responses, int):
            raise TypeError("num_responses must be an integer")
        if num_responses < 1:
            raise ValueError("num_responses must be positive")
        scorer_names = tuple(self.SCORER_DEFAULTS if scorers is None else scorers)
        if not scorer_names or len(set(scorer_names)) != len(scorer_names):
            raise ValueError("scorers must be a non-empty list of unique names")
        unknown = set(scorer_names) - set(SCORER_FULL)
        if unknown:
            raise ValueError(f"unknown UQLM scorer(s): {', '.join(sorted(unknown))}")
        if primary_scorer not in scorer_names:
            raise ValueError("primary_scorer must be included in scorers")

        if backend is None:
            try:
                from uqlm import BlackBoxUQ
            except ImportError as exc:
                raise ImportError(
                    "UQLM scoring requires the optional dependency; "
                    "install it with `pip install 'moira[uq]'`."
                ) from exc
            kwargs: dict[str, Any] = {
                "llm": None,
                "scorers": list(scorer_names),
                "use_best": False,
                "verbose": False,
            }
            if device is not None:
                kwargs["device"] = device
            backend = BlackBoxUQ(**kwargs)

        self.llm = llm
        self.num_responses = num_responses
        self._scorer_names = scorer_names
        self._primary_scorer = primary_scorer
        self._backend = backend

    def score_prompt(self, prompt: str) -> ScoreOutput:
        return self.score_prompt_from_responses(
            generate_responses(self.llm, prompt, self.num_responses)
        )

    def score_prompt_from_responses(
        self,
        raw_responses: Sequence[str],
    ) -> ScoreOutput:
        """Score confidence from pre-generated responses (no extra LLM calls)."""
        answer, agreement = majority_result(raw_responses)
        majority = "YES" if answer else "NO"
        processed = [
            "YES" if decision else "NO"
            for response in raw_responses
            if (decision := parse_yes_no(response)) is not None
        ]
        result = self._backend.score(
            responses=[majority],
            sampled_responses=[processed],
            show_progress_bars=False,
        )
        scores: dict[str, float] = {"exact_match": agreement}
        for key, value in result.data.items():
            if key in self._scorer_names and isinstance(value, list) and value:
                scores[key] = clamp_score(value[0], name=key)
        confidence = scores.get(self._primary_scorer, agreement)
        return answer, confidence, scores
