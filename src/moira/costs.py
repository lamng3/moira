"""What MOIRA's model calls cost: one meter on every chat model, totals for every run.

`METER` is a LangChain callback handler. `attach(llm)` adds it to a chat
model, and from then on every call the model makes (the agent's answers, tool
follow-ups, the alignment veto) is recorded: model, input and output tokens,
seconds, and dollars at list price when the model has one. Local models
(Ollama, Hugging Face) have no per-token price; their cost is the seconds
they ran.

Experiments write `METER.summary()` into their metrics under `cost`, so every
results table can show what a run spent next to how well it did. Prices are
dollars per million tokens (input, output); extend or override them with
MOIRA_PRICES, a JSON object of the same shape.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:  # pragma: no cover - langchain is a core dependency
    BaseCallbackHandler = object  # type: ignore[misc, assignment]

PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
}


def prices() -> dict[str, tuple[float, float]]:
    table = dict(PRICES)
    table.update({k: tuple(v) for k, v in json.loads(os.environ.get("MOIRA_PRICES", "{}")).items()})
    return table


def price(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """List-price dollars for one call, or None when the model has no price (a local model)."""
    rate = prices().get(model.split(":")[-1].split("/")[-1].lower())
    if rate is None:
        return None
    return (input_tokens * rate[0] + output_tokens * rate[1]) / 1e6


def _usage(response: Any) -> tuple[int, int]:
    """Input and output tokens from a LangChain LLMResult, whichever way the provider reports them."""
    for generations in getattr(response, "generations", []) or []:
        for generation in generations:
            message = getattr(generation, "message", None)
            meta = getattr(message, "usage_metadata", None) or {}
            if meta:
                return int(meta.get("input_tokens", 0)), int(meta.get("output_tokens", 0))
            info = getattr(generation, "generation_info", None) or {}
            if "prompt_eval_count" in info or "eval_count" in info:  # Ollama
                return int(info.get("prompt_eval_count", 0)), int(info.get("eval_count", 0))
    usage = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
    return int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))


class UsageMeter(BaseCallbackHandler):  # type: ignore[misc, valid-type]
    """Records every model call it is attached to. Thread-safe."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._started: dict[Any, tuple[float, str]] = {}
        self._lock = threading.Lock()

    def on_chat_model_start(self, serialized: dict, messages: Any, *, run_id: Any, **kwargs: Any) -> None:
        self._start(serialized, run_id, kwargs)

    def on_llm_start(self, serialized: dict, prompts: Any, *, run_id: Any, **kwargs: Any) -> None:
        self._start(serialized, run_id, kwargs)

    def _start(self, serialized: dict, run_id: Any, kwargs: dict) -> None:
        params = kwargs.get("invocation_params") or {}
        metadata = kwargs.get("metadata") or {}
        model = (
            metadata.get("ls_model_name")
            or params.get("model")
            or params.get("model_name")
            or (serialized or {}).get("name", "unknown")
        )
        with self._lock:
            self._started[run_id] = (time.time(), str(model))

    def on_llm_end(self, response: Any, *, run_id: Any, **kwargs: Any) -> None:
        with self._lock:
            started, model = self._started.pop(run_id, (time.time(), "unknown"))
        tokens_in, tokens_out = _usage(response)
        self.record(model, tokens_in, tokens_out, time.time() - started)

    def record(self, model: str, input_tokens: int, output_tokens: int, seconds: float) -> dict[str, Any]:
        call = {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "seconds": round(seconds, 3),
            "cost_usd": price(model, input_tokens, output_tokens),
        }
        with self._lock:
            self.calls.append(call)
        return call

    def reset(self) -> None:
        with self._lock:
            self.calls.clear()

    def summary(self) -> dict[str, Any]:
        """Totals: calls, tokens, model seconds, and dollars for the calls that have a price."""
        with self._lock:
            calls = list(self.calls)
        priced = [c["cost_usd"] for c in calls if c["cost_usd"] is not None]
        return {
            "calls": len(calls),
            "input_tokens": sum(c["input_tokens"] for c in calls),
            "output_tokens": sum(c["output_tokens"] for c in calls),
            "model_seconds": round(sum(c["seconds"] for c in calls), 2),
            "cost_usd": round(sum(priced), 6) if priced else None,
            "unpriced_calls": len(calls) - len(priced),
            "models": sorted({c["model"] for c in calls}),
        }


METER = UsageMeter()


def attach(llm: Any, meter: UsageMeter = METER) -> Any:
    """Add `meter` to a LangChain chat model's callbacks. Returns the model."""
    try:
        callbacks = list(getattr(llm, "callbacks", None) or [])
        if meter not in callbacks:
            llm.callbacks = [*callbacks, meter]
    except (AttributeError, TypeError, ValueError):
        pass  # not a LangChain model: it is simply not metered
    return llm
