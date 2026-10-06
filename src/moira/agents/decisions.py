"""Optional Jev decisions for model routing and tool-risk gating."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from moira.agents.core.tool_executor import redact

Classifier = Callable[[str, Mapping[str, Mapping[str, str]]], Any]

ROUTE_CRITERIA = {
    "local": "Short lookups, extraction, and questions answerable from retrieved ontology context.",
    "standard": "Normal ontology question answering and relation lookup.",
    "careful": "Alignment disputes, conflicts, refinement, or multi-ontology reasoning.",
}
ROUTE_INSTRUCTIONS = "Choose the least costly model that can complete the ontology task."
RISK_INSTRUCTIONS = (
    "This tool call is unsafe to run because it sends ontology terms or private "
    "data to an external service, or requests a destructive action."
)
DEFAULT_GATED_TOOLS = frozenset({"search_term_context", "search_knowledge_graph"})


@dataclass(frozen=True, slots=True)
class ModelRoute:
    """One routing decision for a single agent run."""

    name: str
    model_name: str
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float | None = None
    fail_open: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_name": self.model_name,
            "probabilities": self.probabilities,
            "confidence": self.confidence,
            "fail_open": self.fail_open,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class ToolGateDecision:
    """Whether a parsed tool call may execute."""

    tool_name: str
    gated: bool
    blocked: bool
    risk: float | None = None
    fail_closed: bool = False
    error: str | None = None

    def refusal(self) -> dict[str, Any]:
        reason = self.error or (
            f"Risk probability {self.risk:.2f} meets the configured threshold."
            if self.risk is not None
            else "Tool call blocked."
        )
        return {
            "blocked": True,
            "tool_name": self.tool_name,
            "reason": reason,
            "risk": self.risk,
            "fail_closed": self.fail_closed,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.refusal() if self.blocked else {"blocked": False}
        payload.update(
            {
                "tool_name": self.tool_name,
                "gated": self.gated,
                "risk": self.risk,
                "fail_closed": self.fail_closed,
                "error": self.error,
            }
        )
        return payload


class JevModelRouter:
    """Choose a model once from the user query."""

    def __init__(
        self,
        *,
        classifier: Classifier | None = None,
        model_factory: Callable[[str], Any] | None = None,
        routes: Mapping[str, str] | None = None,
    ) -> None:
        self._classifier = classifier
        self._model_factory = model_factory or _default_model_factory
        self._routes = dict(routes or {})

    def select(
        self, query: str, fallback_model: str
    ) -> tuple[Any | None, ModelRoute]:
        route = self.route(query, fallback_model)
        if route.fail_open or route.model_name == fallback_model:
            return None, route
        try:
            return self._model_factory(route.model_name), route
        except Exception as exc:
            return None, ModelRoute(
                name="standard",
                model_name=fallback_model,
                fail_open=True,
                error=str(exc),
            )

    def route(self, query: str, fallback_model: str) -> ModelRoute:
        models = _route_models(fallback_model, self._routes)
        try:
            if not query.strip():
                raise ValueError("query must not be empty")
            response = self._classify(
                query,
                {
                    "route": {
                        "type": "choice",
                        "instructions": ROUTE_INSTRUCTIONS,
                        "criteria": ROUTE_CRITERIA,
                    }
                },
            )
            choice = _choice_answer(response)
            name = str(choice.get("choice") or "")
            if name not in models:
                raise ValueError(f"Unknown model route {name!r}")
            return ModelRoute(
                name=name,
                model_name=models[name],
                probabilities=_float_map(choice.get("probabilities")),
                confidence=_optional_float(choice.get("confidence")),
            )
        except Exception as exc:
            return ModelRoute(
                name="standard",
                model_name=fallback_model,
                fail_open=True,
                error=str(exc),
            )

    def _classify(self, state: str, questions: Mapping[str, Mapping[str, str]]) -> Any:
        if self._classifier is not None:
            return self._classifier(state, questions)
        return _live_classify(state, questions)


class ToolRiskGate:
    """Block configured tool calls whose Jev risk probability is too high."""

    def __init__(
        self,
        *,
        classifier: Classifier | None = None,
        tools: Sequence[str] | None = None,
        threshold: float = 0.5,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self._classifier = classifier
        self.tools = frozenset(tools or DEFAULT_GATED_TOOLS)
        self.threshold = threshold

    def assess(self, tool_call: Mapping[str, Any]) -> ToolGateDecision:
        name = str(tool_call.get("tool_name") or "")
        if name not in self.tools:
            return ToolGateDecision(tool_name=name, gated=False, blocked=False)
        redacted = redact(dict(tool_call.get("arguments") or {}))
        state = json.dumps(
            {"tool_name": name, "arguments": redacted},
            sort_keys=True,
            default=str,
        )
        try:
            response = self._classify(
                state,
                {"risky": {"type": "noul", "instructions": RISK_INSTRUCTIONS}},
            )
            risk = _noul_answer(response)
            return ToolGateDecision(
                tool_name=name,
                gated=True,
                blocked=risk >= self.threshold,
                risk=risk,
            )
        except Exception as exc:
            return ToolGateDecision(
                tool_name=name,
                gated=True,
                blocked=True,
                fail_closed=True,
                error=str(exc),
            )

    def _classify(self, state: str, questions: Mapping[str, Mapping[str, str]]) -> Any:
        if self._classifier is not None:
            return self._classifier(state, questions)
        return _live_classify(state, questions)


def create_model_router(
    *,
    classifier: Classifier | None = None,
    model_factory: Callable[[str], Any] | None = None,
) -> JevModelRouter | None:
    """Return a router when ``MOIRA_MODEL_ROUTING=jev``."""
    if os.getenv("MOIRA_MODEL_ROUTING", "").strip().lower() != "jev":
        return None
    return JevModelRouter(classifier=classifier, model_factory=model_factory)


def create_tool_gate(
    *,
    classifier: Classifier | None = None,
) -> ToolRiskGate | None:
    """Return a gate when ``MOIRA_TOOL_GATE=jev``."""
    if os.getenv("MOIRA_TOOL_GATE", "").strip().lower() != "jev":
        return None
    raw_tools = os.getenv("MOIRA_GATED_TOOLS", "").strip()
    tools = (
        [item.strip() for item in raw_tools.split(",") if item.strip()]
        if raw_tools
        else None
    )
    raw_threshold = os.getenv("MOIRA_TOOL_RISK_THRESHOLD", "").strip()
    threshold = float(raw_threshold) if raw_threshold else 0.5
    return ToolRiskGate(classifier=classifier, tools=tools, threshold=threshold)


def _route_models(fallback_model: str, overrides: Mapping[str, str]) -> dict[str, str]:
    models = {
        "local": os.getenv("MOIRA_ROUTE_LOCAL", "ollama:phi3"),
        "standard": os.getenv("MOIRA_ROUTE_STANDARD", fallback_model),
        "careful": os.getenv("MOIRA_ROUTE_CAREFUL", fallback_model),
    }
    models.update({key: value for key, value in overrides.items() if value})
    return models


def _default_model_factory(model_name: str) -> Any:
    from moira.model_runtime import create_model_runtime

    return create_model_runtime(model_name=model_name).llm


def _export_api_key() -> None:
    if os.environ.get("TYPESAFE_API_KEY"):
        return
    key = os.environ.get("JEV_API_KEY", "").strip()
    if not key:
        raise RuntimeError("JEV_API_KEY or TYPESAFE_API_KEY is required")
    os.environ["TYPESAFE_API_KEY"] = key


def _live_classify(state: str, questions: Mapping[str, Mapping[str, Any]]) -> Any:
    _export_api_key()
    try:
        from langchain_typesafe import Choice, Noul, TypeSafeClassifier
    except ImportError as exc:
        raise RuntimeError(
            "Jev decisions require the 'jev' extra: uv sync --extra jev"
        ) from exc

    converted = {}
    for name, spec in questions.items():
        if spec.get("type") == "noul":
            converted[name] = Noul(instructions=str(spec["instructions"]))
        else:
            converted[name] = Choice(
                instructions=str(spec["instructions"]),
                criteria=dict(spec.get("criteria") or {}),
            )
    return TypeSafeClassifier().invoke({"state": state, "questions": converted})


def _choice_answer(response: Any) -> Mapping[str, Any]:
    if isinstance(response, Mapping):
        return response
    choices = getattr(response, "choices", None)
    answer = choices.get("route") if isinstance(choices, Mapping) else None
    if answer is None:
        raise ValueError("Jev response did not include a route choice")
    if isinstance(answer, Mapping):
        return answer
    return {
        "choice": getattr(answer, "choice", None),
        "probabilities": getattr(answer, "probabilities", {}),
        "confidence": getattr(answer, "confidence", None),
    }


def _noul_answer(response: Any) -> float:
    if isinstance(response, Mapping):
        value = response.get("noul", response.get("risky"))
        if isinstance(value, Mapping):
            value = value.get("noul")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        raise ValueError("Jev response did not include a risk probability")
    nouls = getattr(response, "nouls", None)
    answer = nouls.get("risky") if isinstance(nouls, Mapping) else None
    probability = getattr(answer, "noul", None)
    if isinstance(probability, (int, float)) and not isinstance(probability, bool):
        return float(probability)
    raise ValueError("Jev response did not include a risk probability")


def _float_map(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): float(item)
        for key, item in value.items()
        if isinstance(item, (int, float)) and not isinstance(item, bool)
    }


def _optional_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None
