"""Typed questions over a state, answered by the small router model."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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
CHOICE_LIMIT = 26
DEFAULT_HARNESS_URL = "http://127.0.0.1:30000"
DEFAULT_HARNESS_MODEL = "mlx-community/Qwen3-4B-4bit"


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
    """Questions that decide whether a typed question matches the hot cache.

    The choice stays within 26 options, including retrieve. A longer hot
    cache keeps the most recent questions that fit.
    """
    remembered = [question for question in cached_questions if question]
    criteria = {
        f"q{index}": question
        for index, question in enumerate(remembered[: CHOICE_LIMIT - 1])
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

    def ask_systemone(
        self,
        state: Mapping[str, str] | str,
        questions: Mapping[str, Question],
        base_url: str,
    ) -> HarnessReading:
        """Read Choice, Score, and Noul probabilities from /v1/systemone."""
        if not questions:
            return HarnessReading()
        body = {
            "model": os.getenv("AGENTOI_HARNESS_MODEL", DEFAULT_HARNESS_MODEL)
            or DEFAULT_HARNESS_MODEL,
            "state": state if isinstance(state, str) else dict(state),
            "questions": _systemone_questions(questions),
        }
        try:
            payload = _post_json(base_url.rstrip("/") + "/v1/systemone", body)
        except (
            HTTPError,
            URLError,
            TimeoutError,
            OSError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            return HarnessReading(fail_open=True, error=str(exc))
        raw_answers = payload.get("answers") if isinstance(payload, dict) else None
        if not isinstance(raw_answers, dict):
            return HarnessReading(fail_open=True, error="systemone response had no answers")
        return HarnessReading(
            answers={
                name: _answer(question, raw_answers.get(name))
                for name, question in questions.items()
            }
        )


def _systemone_questions(questions: Mapping[str, Question]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for name, question in questions.items():
        if isinstance(question, Choice):
            options = list(question.criteria.items())[:CHOICE_LIMIT]
            payload[name] = {
                "type": "choice",
                "instructions": question.instructions,
                "criteria": {key: description or None for key, description in options},
            }
        elif isinstance(question, Score):
            payload[name] = {
                "type": "score",
                "instructions": question.instructions,
                "criteria": list(question.criteria),
            }
        else:
            payload[name] = {"type": "noul", "instructions": question.instructions}
    return payload


def _post_json(url: str, body: Mapping[str, object]) -> object:
    request = Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode())
    except HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise ValueError(detail or str(exc)) from exc


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
            lines.append("  Reply with the option key, a probability for each option, and a confidence.")
            for key, description in question.criteria.items():
                lines.append(f"  {key}: {description}")
            shape[name] = {
                "choice": "retrieve",
                "probabilities": {"retrieve": 1.0},
                "confidence": 0.0,
            }
        elif isinstance(question, Score):
            labels = ", ".join(question.criteria)
            lines.append(
                "  Reply with one level, a probability for each level, and a confidence."
            )
            lines.append(f"  Levels: {labels}.")
            first = question.criteria[0] if question.criteria else ""
            shape[name] = {
                "score": first,
                "probabilities": {first: 1.0} if first else {},
                "confidence": 0.0,
            }
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
    text = chosen.strip().casefold()
    folded = {key.casefold(): key for key in criteria}
    direct = folded.get(text)
    if direct:
        return direct
    for key, description in criteria.items():
        if description.casefold() == text:
            return key
    return None


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
