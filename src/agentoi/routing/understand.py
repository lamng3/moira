"""Small understanding model that chooses replay or retrieve."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agentoi.routing.harness import (
    Choice,
    ChoiceAnswer,
    Harness,
    NoulAnswer,
    ScoreAnswer,
    ontology_questions,
)

DEFAULT_ROUTER_MODEL = "ollama:phi3"


def router_model_name() -> str:
    """Small model used only to name the next action."""
    return os.getenv("AGENTOI_ROUTER_MODEL", DEFAULT_ROUTER_MODEL) or DEFAULT_ROUTER_MODEL


@dataclass(frozen=True, slots=True)
class ActionRoute:
    """One understanding decision for a question."""

    action: str
    question: str | None = None
    fail_open: bool = False
    error: str | None = None
    same_intent: float | None = None
    closeness: str | None = None

    def note(self) -> str:
        """Run-log line for a replay, including the other typed answers."""
        text = f"Routed to a remembered question: {self.question}."
        details: list[str] = []
        if self.same_intent is not None:
            details.append(f"same intent {self.same_intent:.2f}")
        if self.closeness:
            details.append(f"closeness {self.closeness}")
        if not details:
            return text
        return f"{text} ({', '.join(details)})."


def understand_question(
    question: str,
    cached_questions: Sequence[str],
    llm: Any,
) -> ActionRoute:
    """Ask the Harness whether this question matches one in the cache.

    A failed call falls open to retrieve. Replay is accepted only when the
    choice names a question that is actually cached.
    """
    cached = [item for item in cached_questions if item]
    if not cached:
        return ActionRoute("retrieve")
    questions = ontology_questions(cached)
    reading = Harness().ask({"question": question}, questions, llm)
    same_intent = _same_intent(reading.answers.get("same_intent"))
    closeness = _closeness(reading.answers.get("closeness"))
    if reading.fail_open:
        return ActionRoute(
            "retrieve",
            fail_open=True,
            error=reading.error,
            same_intent=same_intent,
            closeness=closeness,
        )
    match = questions["match"]
    choice = reading.answers.get("match")
    key = choice.choice if isinstance(choice, ChoiceAnswer) else None
    named = match.criteria.get(key) if isinstance(match, Choice) and key else None
    if key and key != "retrieve" and isinstance(named, str) and named in cached:
        return ActionRoute(
            "replay",
            question=named,
            same_intent=same_intent,
            closeness=closeness,
        )
    return ActionRoute("retrieve", same_intent=same_intent, closeness=closeness)


def _same_intent(answer: object) -> float | None:
    if isinstance(answer, NoulAnswer):
        return answer.noul
    return None


def _closeness(answer: object) -> str | None:
    if isinstance(answer, ScoreAnswer):
        return answer.label
    return None
