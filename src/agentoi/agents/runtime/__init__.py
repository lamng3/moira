"""Runtime tracing and persistence primitives."""

from .trace import Trace, TraceStep, export_trace

__all__ = [
    "Trace",
    "TraceStep",
    "export_trace",
    "save_run_graph",
    "save_trace_to_file",
]


def __getattr__(name: str):
    if name in {"save_run_graph", "save_trace_to_file"}:
        from .persistence import save_run_graph, save_trace_to_file

        return {
            "save_run_graph": save_run_graph,
            "save_trace_to_file": save_trace_to_file,
        }[name]
    raise AttributeError(name)
