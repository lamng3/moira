"""Shared dependency-free types and utilities for uncertainty scoring."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Protocol, Sequence, TypeAlias, runtime_checkable

ScoreOutput: TypeAlias = tuple[bool, float, dict[str, float]]


@runtime_checkable
class ResponseScorer(Protocol):
    def score_prompt_from_responses(
        self, raw_responses: Sequence[str]
    ) -> ScoreOutput: ...


def parse_yes_no(text: str) -> bool | None:
    """Extract an unambiguous YES/NO decision without substring matching."""

    if not isinstance(text, str):
        return None
    tagged = re.search(
        r"<Answer>\s*(YES|NO)\s*</Answer>", text, flags=re.IGNORECASE
    )
    if tagged:
        return tagged.group(1).upper() == "YES"
    json_answer = re.search(
        r"""["']answer["']\s*:\s*["'](yes|no)["']""",
        text,
        flags=re.IGNORECASE,
    )
    if json_answer:
        return json_answer.group(1).upper() == "YES"
    tokens = re.findall(r"\b(?:YES|NO)\b", text, flags=re.IGNORECASE)
    decisions = {token.upper() for token in tokens}
    if len(decisions) != 1:
        return None
    return next(iter(decisions)) == "YES"


def majority_result(raw_responses: Sequence[str]) -> tuple[bool, float]:
    if not raw_responses:
        raise ValueError("at least one response is required")
    decisions = [decision for text in raw_responses if (decision := parse_yes_no(text)) is not None]
    if not decisions:
        raise ValueError("responses contain no parseable YES/NO decisions")
    yes_count = sum(decisions)
    no_count = len(decisions) - yes_count
    answer = yes_count > no_count
    agreement = max(yes_count, no_count) / len(decisions)
    return answer, agreement


def generate_responses(llm: Any, prompt: str, count: int) -> list[str]:
    if llm is None or not callable(getattr(llm, "invoke", None)):
        raise TypeError("llm must provide an invoke(prompt) method")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must not be empty")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("num_responses must be a positive integer")
    responses: list[str] = []
    for _ in range(count):
        response = llm.invoke(prompt)
        content = getattr(response, "content", None)
        if content is None:
            content = getattr(response, "resp", None)
        responses.append(str(response if content is None else content).strip())
    return responses


def clamp_score(value: object, *, name: str = "score") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"{name} must be finite")
    return min(1.0, max(0.0, score))


def validated_scores(scores: Mapping[str, object]) -> dict[str, float]:
    return {name: clamp_score(value, name=name) for name, value in scores.items()}
