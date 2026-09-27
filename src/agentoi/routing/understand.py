"""Small understanding model that chooses replay or retrieve."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agentoi.memory.path import normalize_question

DEFAULT_ROUTER_MODEL = "ollama:phi3"
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


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


def understand_question(
    question: str,
    cached_questions: Sequence[str],
    llm: Any,
) -> ActionRoute:
    """Cross-check a question against cached questions, then name an action.

    A failed call falls open to retrieve. Replay is accepted only when the
    named question is one of the cached questions.
    """
    cached = [item for item in cached_questions if item]
    if not cached:
        return ActionRoute("retrieve")
    try:
        response = llm.invoke(_prompt(question, cached))
    except Exception as exc:
        return ActionRoute("retrieve", fail_open=True, error=str(exc))
    text = getattr(response, "content", str(response))
    return _parse(text, cached)


def _prompt(question: str, cached: Sequence[str]) -> str:
    lines = [
        "You route an ontology question. Reply with one JSON object and nothing else.",
        'Use {"action":"replay","question":"<one cached question>"} when the new question asks the same thing as a cached question.',
        'Use {"action":"retrieve"} when none of the cached questions match.',
        "Cached questions:",
    ]
    lines.extend(f"- {item}" for item in cached)
    lines.append(f"New question: {question}")
    return "\n".join(lines)


def _parse(text: str, cached: Sequence[str]) -> ActionRoute:
    match = _JSON_OBJECT.search(text or "")
    if match is None:
        return ActionRoute("retrieve")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return ActionRoute("retrieve")
    if not isinstance(payload, dict):
        return ActionRoute("retrieve")
    action = payload.get("action")
    if action != "replay":
        return ActionRoute("retrieve")
    named = payload.get("question")
    if not isinstance(named, str):
        return ActionRoute("retrieve")
    key = normalize_question(named)
    by_key = {normalize_question(item): item for item in cached}
    cached_question = by_key.get(key)
    if cached_question is None:
        return ActionRoute("retrieve")
    return ActionRoute("replay", question=cached_question)
