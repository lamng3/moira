"""Small understanding model that chooses replay or retrieve."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agentoi.routing.harness import Harness, Noul, ontology_questions

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
    continues: bool = False

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
    turns: Sequence[tuple[str, str]] | None = None,
) -> ActionRoute:
    """Ask the Harness once, then let the local policy choose replay or retrieve.

    A failed call falls open to retrieve. Prior turns are the last few
    question and answer pairs from this chat.
    """
    from agentoi.routing.policy import RoutePolicy

    cached = [item for item in cached_questions if item]
    recent = list(turns or [])[-4:]
    questions = ontology_questions(cached) if cached else {}
    if recent:
        questions["continues"] = Noul(
            instructions=(
                "Does this question continue the previous turn in the chat, "
                "rather than start a new topic?"
            )
        )
    if not questions:
        return ActionRoute("retrieve")
    state = {"question": question}
    if recent:
        state["chat"] = "\n".join(f"Q: {prior}\nA: {answer}" for prior, answer in recent)
    reading = Harness().ask(state, questions, llm)
    return RoutePolicy().decide(reading, questions, cached)
