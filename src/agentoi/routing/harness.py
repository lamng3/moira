"""Typed questions over a state, answered by the small router model."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True, slots=True)
class Noul:
    """A yes-or-no question. The answer is the probability the statement is true."""

    instructions: str


@dataclass(frozen=True, slots=True)
class Choice:
    """Pick one named option. Each criterion is a key and its description."""

    instructions: str
    criteria: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class Score:
    """Place the state on an ordered scale. The first criterion is the low end."""

    instructions: str
    criteria: Sequence[str]


Question = Noul | Choice | Score


@dataclass(frozen=True, slots=True)
class NoulAnswer:
    noul: float | None = None


@dataclass(frozen=True, slots=True)
class ChoiceAnswer:
    choice: str | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class ScoreAnswer:
    score: float | None = None
    label: str | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float | None = None


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


@dataclass(frozen=True, slots=True)
class HarnessReading:
    """Answers to one set of typed questions."""

    answers: dict[str, Answer] = field(default_factory=dict)
    fail_open: bool = False
    error: str | None = None


def ontology_questions(cached_questions: Sequence[str]) -> dict[str, Question]:
    """Questions that decide whether a typed question matches the hot cache."""
    criteria = {
        f"q{index}": question
        for index, question in enumerate(cached_questions)
        if question
    }
    criteria["retrieve"] = "None of these questions match. Look up concepts."
    return {
        "same_intent": Noul(
            instructions="Does this question ask the same thing as one of the cached questions?"
        ),
        "match": Choice(
            instructions="Which cached question is the same intent? Choose retrieve if none match.",
            criteria=criteria,
        ),
        "closeness": Score(
            instructions="How close is this question to the nearest cached question?",
            criteria=("Different", "Related", "Same question"),
        ),
    }


class Harness:
    """Ask typed questions about a state and parse the model's JSON."""

    def ask(
        self,
        state: Mapping[str, str] | str,
        questions: Mapping[str, Question],
        llm: Any,
    ) -> HarnessReading:
        if not questions:
            return HarnessReading()
        try:
            response = llm.invoke(_prompt(state, questions))
        except Exception as exc:
            return HarnessReading(fail_open=True, error=str(exc))
        text = getattr(response, "content", str(response))
        return _parse(text, questions)


def _prompt(state: Mapping[str, str] | str, questions: Mapping[str, Question]) -> str:
    lines = [
        "Answer each typed question about the state. Reply with one JSON object and nothing else.",
        "State:",
        _render_state(state),
        "Questions:",
    ]
    shape: dict[str, object] = {}
    for name, question in questions.items():
        lines.append(f"- {name}: {question.instructions}")
        if isinstance(question, Choice):
            lines.append("  Reply with one option key.")
            for key, description in question.criteria.items():
                lines.append(f"  {key}: {description}")
            shape[name] = {"choice": "retrieve"}
        elif isinstance(question, Score):
            labels = ", ".join(question.criteria)
            lines.append(f"  Reply with one of: {labels}.")
            shape[name] = {"score": question.criteria[0] if question.criteria else ""}
        else:
            lines.append("  Reply with a probability from 0 to 1 that the statement is true.")
            shape[name] = {"noul": 0.0}
    lines.append(f"JSON shape: {json.dumps(shape)}")
    return "\n".join(lines)


def _render_state(state: Mapping[str, str] | str) -> str:
    if isinstance(state, str):
        return state
    return "\n".join(f"{key}: {value}" for key, value in state.items())


def _parse(text: str, questions: Mapping[str, Question]) -> HarnessReading:
    match = _JSON_OBJECT.search(text or "")
    if match is None:
        return HarnessReading()
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return HarnessReading()
    if not isinstance(payload, dict):
        return HarnessReading()
    answers: dict[str, Answer] = {}
    for name, question in questions.items():
        answers[name] = _answer(question, payload.get(name))
    return HarnessReading(answers=answers)


def _answer(question: Question, raw: object) -> Answer:
    if isinstance(question, Noul):
        return NoulAnswer(noul=_noul(raw))
    if isinstance(question, Choice):
        return _choice(question, raw)
    return _score(question, raw)


def _noul(raw: object) -> float | None:
    value = raw.get("noul") if isinstance(raw, dict) else raw
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return min(1.0, max(0.0, float(value)))
    return None


def _choice(question: Choice, raw: object) -> ChoiceAnswer:
    if isinstance(raw, str):
        chosen = raw
        probabilities: dict[str, float] = {}
        confidence = None
    elif isinstance(raw, dict):
        chosen = raw.get("choice")
        probabilities = _float_map(raw.get("probabilities"))
        confidence = _optional_float(raw.get("confidence"))
    else:
        return ChoiceAnswer()
    if not isinstance(chosen, str):
        return ChoiceAnswer(probabilities=probabilities, confidence=confidence)
    key = _match_key(chosen, question.criteria)
    return ChoiceAnswer(choice=key, probabilities=probabilities, confidence=confidence)


def _score(question: Score, raw: object) -> ScoreAnswer:
    labels = list(question.criteria)
    if isinstance(raw, dict):
        value = raw.get("score", raw.get("label"))
        probabilities = _float_map(raw.get("probabilities"))
        confidence = _optional_float(raw.get("confidence"))
    else:
        value = raw
        probabilities = {}
        confidence = None
    label, score = _score_label(value, labels)
    return ScoreAnswer(
        score=score,
        label=label,
        probabilities=probabilities,
        confidence=confidence,
    )


def _score_label(value: object, labels: list[str]) -> tuple[str | None, float | None]:
    if isinstance(value, str):
        folded = {label.casefold(): label for label in labels}
        label = folded.get(value.strip().casefold())
        if label is None:
            return None, None
        return label, float(labels.index(label))
    if isinstance(value, (int, float)) and not isinstance(value, bool) and labels:
        index = int(round(float(value)))
        index = min(len(labels) - 1, max(0, index))
        return labels[index], float(value)
    return None, None


def _match_key(chosen: str, criteria: Mapping[str, str]) -> str | None:
    folded = {key.casefold(): key for key in criteria}
    return folded.get(chosen.strip().casefold())


def _float_map(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): float(item)
        for key, item in value.items()
        if isinstance(item, (int, float)) and not isinstance(item, bool)
    }


def _optional_float(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None
