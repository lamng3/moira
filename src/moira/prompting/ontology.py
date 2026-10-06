"""Typed prompts and responses for ontology equivalence decisions."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class PromptResult:
    """A binary equivalence decision with confidence on a 0–1 scale."""

    equivalent: bool
    confidence: float

    def __post_init__(self) -> None:
        if isinstance(self.confidence, bool) or not isinstance(
            self.confidence, (int, float)
        ):
            raise TypeError("confidence must be a number")
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True)
class PromptExample:
    source: str
    target: str
    result: PromptResult

    def __post_init__(self) -> None:
        if not self.source.strip() or not self.target.strip():
            raise ValueError("example source and target must not be empty")


@dataclass(frozen=True)
class PromptTemplate:
    question: str = "Are these ontology classes semantically equivalent?"
    output_instruction: str = (
        'Return only JSON: {"answer":"yes"|"no","confidence":0.0-1.0}'
    )

    def __post_init__(self) -> None:
        if not self.question.strip() or not self.output_instruction.strip():
            raise ValueError("template fields must not be empty")


@dataclass(frozen=True)
class PromptConfig:
    instruction: str = (
        "Decide whether the two ontology classes denote the same concept."
    )
    examples: tuple[PromptExample, ...] = field(default_factory=tuple)
    template: PromptTemplate = field(default_factory=PromptTemplate)

    def __post_init__(self) -> None:
        if not self.instruction.strip():
            raise ValueError("instruction must not be empty")
        object.__setattr__(self, "examples", tuple(self.examples))


class OntologyEquivalencePromptBuilder:
    """Build compact, machine-readable ontology equivalence prompts."""

    def __init__(self, config: PromptConfig | None = None) -> None:
        self.config = config or PromptConfig()

    def build(self, source: object, target: object) -> str:
        sections = [self.config.instruction.strip()]
        sections.extend(self._format_example(example) for example in self.config.examples)
        sections.append(self._format_pair(_describe(source), _describe(target)))
        return "\n\n".join(sections)

    def _format_example(self, example: PromptExample) -> str:
        answer = "yes" if example.result.equivalent else "no"
        response = json.dumps(
            {"answer": answer, "confidence": example.result.confidence},
            separators=(",", ":"),
        )
        return f"{self._format_pair(example.source, example.target)}\n{response}"

    def _format_pair(self, source: str, target: str) -> str:
        template = self.config.template
        return (
            f"{template.question}\n"
            f"A: {source.strip()}\n"
            f"B: {target.strip()}\n"
            f"{template.output_instruction}"
        )


def parse_equivalence_answer(text: str) -> PromptResult:
    """Parse a JSON answer, allowing a fenced block or surrounding model text."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("response must not be empty")

    errors: list[str] = []
    for candidate in _json_objects(text):
        try:
            payload = json.loads(candidate)
            return _result_from_mapping(payload)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            errors.append(str(exc))

    detail = f": {errors[-1]}" if errors else ""
    raise ValueError(f"response does not contain a valid equivalence result{detail}")


def _result_from_mapping(payload: Any) -> PromptResult:
    if not isinstance(payload, Mapping):
        raise ValueError("result must be a JSON object")
    if "answer" not in payload or "confidence" not in payload:
        raise ValueError("result requires answer and confidence")
    answer = payload["answer"]
    if not isinstance(answer, str) or answer.strip().lower() not in {"yes", "no"}:
        raise ValueError("answer must be 'yes' or 'no'")
    confidence = payload["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be a JSON number")
    return PromptResult(answer.strip().lower() == "yes", float(confidence))


def _json_objects(text: str) -> Iterable[str]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            _, end = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        yield text[match.start() : match.start() + end]


def _describe(value: object) -> str:
    if isinstance(value, str):
        description = value
    else:
        describe = getattr(value, "describe", None)
        if not callable(describe):
            raise TypeError("ontology classes must be strings or provide describe()")
        description = describe()
    if not isinstance(description, str) or not description.strip():
        raise ValueError("ontology class description must not be empty")
    return description
