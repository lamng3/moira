"""Run an MOIRA agent with a local fake model and tool."""

from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path

from moira.agents import Agent, OOTAgentObserver


def echo(text: str, uppercase: bool = False) -> dict:
    """Return text and its length."""
    value = text.upper() if uppercase else text
    return {"text": value, "length": len(value)}


class FakeLLM:
    """Return one deterministic tool call."""

    model = "fake-model"

    def invoke(self, messages):
        call = {
            "tool_name": "echo",
            "tool_type": "module",
            "module_path": "example_tools",
            "arguments": {"text": "hello ontology", "uppercase": True},
        }
        return types.SimpleNamespace(content=f"```json\n{json.dumps(call)}\n```")


def main() -> None:
    module = types.ModuleType("example_tools")
    module.echo = echo
    sys.modules[module.__name__] = module

    registry = {
        "echo": {
            "tool_name": "echo",
            "tool_type": "module",
            "module_path": module.__name__,
            "arguments": {},
        }
    }

    with tempfile.TemporaryDirectory() as directory:
        tools_path = Path(directory) / "tools.json"
        tools_path.write_text(json.dumps(registry), encoding="utf-8")
        agent = Agent(
            llm=FakeLLM(),
            tools_path=tools_path,
            observers=[],
            oot_observer=OOTAgentObserver(save_traces=False),
            verbose=False,
            save_traces=False,
            save_local=False,
        )
        result = agent.invoke("Echo a greeting.", return_trace=True)

    print(json.dumps(result["result"], indent=2))


if __name__ == "__main__":
    main()
