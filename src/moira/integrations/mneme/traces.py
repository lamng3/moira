"""MOIRA agent runs as Mneme traces: live, from the agent's events, or imported from saved trace files.

`MnemeTraceObserver` is an agent observer, like `OOTAgentObserver`: pass it
to an agent and every run is written to a shared Mneme store as it happens,
one memory per step, so other agents (KROMA's matcher, a defender) can read
MOIRA's reasoning and tool results as premises while the run is still going.

| MOIRA event | Mneme step |
|:--|:--|
| `trace.start` | trace root (task = the initial prompt, run = `run_id`) |
| `model.routed` | `routing` |
| `llm.request` | `llm_request` |
| `llm.response` | `thought` |
| `tool.parsed` | `action` |
| `tool.result` | `observation`, whose parent is the action it answers |
| `tool.blocked` | `error` |
| `trace.end` | `result` (if any) and the end of the trace |

`import_trace` reads a saved `memory_trace_*.json` (the Ontology-of-Thought
`MemoryTrace`) into the same shape, keeping each node's parent, so traces
recorded before the store existed can be compacted, recalled, and audited.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mneme import SHARED, MemoryStore, Trace

AGENT = "moira"

# MemoryTrace node type -> Mneme step kind.
NODE_KINDS = {
    "prompt": "message",
    "reasoning": "llm_request",
    "response": "thought",
    "tool_call": "action",
    "tool_result": "observation",
    "reflection": "reflection",
}


class MnemeTraceObserver:
    """Write each agent run to `store` as a trace, step by step."""

    def __init__(self, store: MemoryStore, agent: str = AGENT, scope: str = SHARED) -> None:
        self.store = store
        self.agent = agent
        self.scope = scope
        self.trace: Trace | None = None
        self.completed: list[Trace] = []
        self._last_action: str | None = None

    def __call__(self, evt: dict[str, Any]) -> None:
        self.on_event(evt)

    def on_event(self, evt: dict[str, Any]) -> None:
        event = evt.get("event")
        if event == "trace.start":
            task = evt.get("initial_prompt") or ""
            self.trace = Trace.start(self.store, self.agent, task, run=evt.get("run_id"), scope=self.scope)
            self._last_action = None
            return
        t = self.trace
        if t is None:
            return
        if event == "model.routed":
            t.step("routing", {"name": evt.get("name"), "model": evt.get("model_name")})
        elif event == "llm.request":
            t.step("llm_request", {"model": evt.get("model"), "prompt": evt.get("message_preview", "")})
        elif event == "llm.response":
            t.thought(evt.get("preview", "") or "")
        elif event == "tool.parsed":
            call = evt.get("tool_call") or {}
            self._last_action = t.action(call.get("tool_name", "unknown"), call.get("arguments")).id
        elif event == "tool.result":
            content = {"tool": evt.get("tool_name"), "content": evt.get("preview", "")}
            t.observation(content, parent=self._last_action)
        elif event == "tool.blocked":
            decision = evt.get("decision") or {}
            t.error(f"blocked {evt.get('tool_name', 'unknown')}: {decision.get('reason', '')}")
        elif event == "trace.end":
            if evt.get("result_preview"):
                t.result(evt["result_preview"])
            t.end("error" if evt.get("error") else "ok", **({"error": evt["error"]} if evt.get("error") else {}))
            self.completed.append(t)
            self.trace = None


def import_trace(
    store: MemoryStore, source: str | Path | dict[str, Any], agent: str = AGENT, scope: str = SHARED
) -> Trace:
    """Read a saved MemoryTrace (a path or its JSON) into `store` as a Mneme trace."""
    data = source if isinstance(source, dict) else json.loads(Path(source).read_text(encoding="utf-8"))
    nodes = data["nodes"]
    root_id = data.get("root_node_id") or next(n for n, v in nodes.items() if not v.get("parent_id"))
    root = nodes[root_id]
    t = Trace.start(store, agent, root.get("content", ""), run=data["id"], scope=scope)
    written: dict[str, str] = {}  # MemoryTrace node id -> Mneme memory id

    def visit(node_id: str, parent: str | None) -> None:
        node = nodes[node_id]
        if node_id != root_id:
            event = (node.get("metadata") or {}).get("event")
            body = {
                "content": node.get("content"),
                "event": event,
                "confidence": node.get("confidence"),
                "tools_used": node.get("tools_used", []),
                "timestamp": node.get("timestamp"),
                "node": node_id,
            }
            kind = "trace_end_note" if event == "trace.end" else NODE_KINDS.get(node.get("type"), node.get("type"))
            written[node_id] = t.step(kind, body, parent=parent).id
            parent = written[node_id]
        for child in node.get("children_ids", []):
            if child in nodes:
                visit(child, parent)

    visit(root_id, None)
    t.end("ok", source="memory_trace")
    return t
