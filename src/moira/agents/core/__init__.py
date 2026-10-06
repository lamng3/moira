"""Core agent orchestration and tool contracts."""

from .events import Event, Observer, emit_event
from .tool_executor import (
    FALLBACK_MODULES,
    coerce_tool_call,
    execute_tool_call,
    extract_tool_json,
    load_tools_json,
    resolve_spec_for,
)

__all__ = [
    "Agent",
    "Event",
    "Observer",
    "emit_event",
    "FALLBACK_MODULES",
    "coerce_tool_call",
    "execute_tool_call",
    "extract_tool_json",
    "load_tools_json",
    "resolve_spec_for",
]


def __getattr__(name: str):
    if name == "Agent":
        from .agent import Agent

        return Agent
    raise AttributeError(name)
