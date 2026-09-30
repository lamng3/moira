"""Local decision over one Harness reading. No extra model call."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from agentoi.routing.harness import Choice, ChoiceAnswer, HarnessReading, NoulAnswer, ScoreAnswer
from agentoi.routing.understand import ActionRoute

SAME_INTENT = 0.8
CONTINUES = 0.8


class RoutePolicy:
    """Replay a cached question, or retrieve. Continuation is a separate flag."""

    def decide(
        self,
        reading: HarnessReading,
        questions: Mapping[str, object],
        cached: Sequence[str],
    ) -> ActionRoute:
        same_intent = _noul(reading.answers.get("same_intent"))
        closeness = _label(reading.answers.get("closeness"))
        continues = (_noul(reading.answers.get("continues")) or 0.0) >= CONTINUES
        if reading.fail_open:
            return ActionRoute(
                "retrieve",
                fail_open=True,
                error=reading.error,
                same_intent=same_intent,
                closeness=closeness,
                continues=continues,
            )
        named = _named_choice(questions.get("match"), reading.answers.get("match"), cached)
        if named:
            return ActionRoute(
                "replay",
                question=named,
                same_intent=same_intent,
                closeness=closeness,
                continues=continues,
            )
        if (same_intent or 0.0) >= SAME_INTENT and closeness == "Same question":
            nearest = _nearest_choice(questions.get("match"), reading.answers.get("match"), cached)
            if nearest:
                return ActionRoute(
                    "replay",
                    question=nearest,
                    same_intent=same_intent,
                    closeness=closeness,
                    continues=continues,
                )
        return ActionRoute(
            "retrieve",
            same_intent=same_intent,
            closeness=closeness,
            continues=continues,
        )


def _named_choice(question: object, answer: object, cached: Sequence[str]) -> str | None:
    if not isinstance(question, Choice) or not isinstance(answer, ChoiceAnswer):
        return None
    if not answer.choice or answer.choice == "retrieve":
        return None
    named = question.criteria.get(answer.choice)
    if isinstance(named, str) and named in cached:
        return named
    return None


def _nearest_choice(question: object, answer: object, cached: Sequence[str]) -> str | None:
    if not isinstance(question, Choice):
        return None
    best_name: str | None = None
    best_probability = -1.0
    if isinstance(answer, ChoiceAnswer):
        for key, probability in answer.probabilities.items():
            named = question.criteria.get(key)
            if not isinstance(named, str) or named not in cached or probability <= best_probability:
                continue
            best_name = named
            best_probability = probability
    if best_name:
        return best_name
    remembered = [item for item in cached if item]
    if len(remembered) == 1:
        return remembered[0]
    return None


def _noul(answer: object) -> float | None:
    if isinstance(answer, NoulAnswer):
        return answer.noul
    return None


def _label(answer: object) -> str | None:
    if isinstance(answer, ScoreAnswer):
        return answer.label
    return None
