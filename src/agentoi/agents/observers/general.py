from typing import Any, Dict

from agentoi.agents.runtime.trace import Trace, TraceStep

def format_event(evt: Dict[str, Any]) -> str | None:
    """Return one terminal line for an agent event."""
    event_type = evt.get("event")
    if event_type == "trace.start":
        return f"▶ Run {evt.get('run_id')} started"
    if event_type == "llm.request":
        return f"🧠 LLM request → {evt.get('model')}: {evt.get('message_preview')}"
    if event_type == "llm.response":
        return f"🧠 LLM response: {evt.get('preview')}"
    if event_type == "tool.parsed":
        tool_call = evt.get("tool_call", {})
        return (
            f"🛠️  Tool parsed: {tool_call.get('tool_name')} "
            f"({tool_call.get('module_path')}) args={tool_call.get('arguments')}"
        )
    if event_type == "tool.exec":
        return f"⚙️  Executing tool: {evt.get('tool_name')} …"
    if event_type == "tool.result":
        return f"✅ Tool result: {evt.get('preview')}"
    if event_type == "tool.skipped":
        return f"⏭️  No tool call ({evt.get('reason')}); returning raw text."
    if event_type == "trace.end":
        if evt.get("error"):
            return f"✖ Run ended with error: {evt['error']}"
        return f"✔ Run finished. Result: {evt.get('result_preview')}"
    return None


def print_observer(evt: Dict[str, Any]):
    line = format_event(evt)
    if line:
        print(line, flush=True)


class LogCapture:
    """Remember the detailed run log so a later command can show it."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, evt: Dict[str, Any]) -> None:
        line = format_event(evt)
        if line:
            self.lines.append(line)

    def clear(self) -> None:
        self.lines.clear()

    def text(self) -> str:
        return "\n".join(self.lines)
