"""MOIRA on a shared Mneme memory: its runs as traces, its query cache as a view, and OntoDB as storage.

MOIRA hosts the workflow; KROMA's matcher and MOIRA's agents share one Mneme
store. This package is how MOIRA talks to that store:

- `traces`: `MnemeTraceObserver` writes each agent run as a trace while it
  runs; `import_trace` loads a saved `memory_trace_*.json`.
- `cache`: `MoiraAnswerCache`, MOIRA's three-tier query cache kept in step with
  the store so it never serves a stale answer, and `MoiraFrontDesk`, the
  read path that asks through it.
- `ontodb`: `OntoDBEngine`, so the memory lives in an OntoDB database next to
  the ontology.

Mneme is optional for MOIRA: install it, or clone it beside this repository.
"""

from __future__ import annotations

from .cache import ANSWER, QUESTION, MoiraAnswerCache, MoiraFrontDesk
from .ontodb import OntoDBEngine, OntoDBError, OntoDBShell
from .traces import MnemeTraceObserver, import_trace

__all__ = [
    "ANSWER",
    "QUESTION",
    "MnemeTraceObserver",
    "MoiraAnswerCache",
    "MoiraFrontDesk",
    "OntoDBEngine",
    "OntoDBError",
    "OntoDBShell",
    "import_trace",
]
