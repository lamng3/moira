from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceStep:
    """Timing and diagnostic data for one agent operation."""

    name: str
    status: str = "init"
    data: dict[str, Any] = field(default_factory=dict)
    start: float = field(default_factory=time.perf_counter)
    end: float | None = None

    @property
    def duration_ms(self) -> int:
        end = self.end if self.end is not None else self.start
        return int((end - self.start) * 1000)


@dataclass
class Trace:
    """Canonical execution trace used by the agent runtime."""

    run_id: str
    steps: list[TraceStep] = field(default_factory=list)
    result_preview: str = ""
    error: str | None = None


def export_trace(trace: Trace) -> dict[str, Any]:
    return {
        "run_id": trace.run_id,
        "steps": [
            {
                "name": step.name,
                "status": step.status,
                "duration_ms": step.duration_ms,
                "data": step.data,
            }
            for step in trace.steps
        ],
        "result_preview": trace.result_preview,
        "error": trace.error,
    }
