import json
import sys
import types

from agentoi.agents import Agent, OOTAgentObserver
from agentoi.agents.runtime import persistence


class FakeLLM:
    model = "fake-model"

    def invoke(self, _messages):
        call = {
            "tool_name": "echo",
            "tool_type": "module",
            "module_path": "agentoi_test_tools",
            "arguments": {"text": "ontology"},
        }
        return types.SimpleNamespace(content=json.dumps(call))


def _agent(tmp_path, **kwargs):
    module = types.ModuleType("agentoi_test_tools")
    module.echo = lambda text: {"text": text, "length": len(text)}
    sys.modules[module.__name__] = module
    registry = tmp_path / "tools.json"
    registry.write_text(
        json.dumps(
            {
                "echo": {
                    "tool_name": "echo",
                    "tool_type": "module",
                    "module_path": module.__name__,
                }
            }
        ),
        encoding="utf-8",
    )
    return Agent(
        FakeLLM(),
        tools_path=registry,
        oot_observer=OOTAgentObserver(save_traces=False),
        verbose=False,
        **kwargs,
    )


def test_plain_answer_does_not_offer_tools(tmp_path):
    agent = _agent(tmp_path, save_traces=False)
    seen: list[list] = []

    class PlainLLM:
        model = "fake-model"

        def invoke(self, messages):
            seen.append(messages)
            return types.SimpleNamespace(content="The heart is an organ.")

    agent.llm = PlainLLM()

    assert agent.answer_plain("QUERY: What is the heart?") == "The heart is an organ."
    blob = " ".join(str(message.content) for message in seen[0])
    assert "Tools list" not in blob
    assert "QUERY: What is the heart?" in blob


def test_agent_invokes_registered_tool_and_returns_oot_trace(tmp_path):
    agent = _agent(tmp_path, save_traces=False)

    response = agent.invoke("Look up ontology.", return_trace=True)

    assert response["result"] == {"text": "ontology", "length": 8}
    assert response["memory_trace"]["id"] == response["trace"]["run_id"]
    assert response["graph"]["thoughts"]
    assert [step["status"] for step in response["trace"]["steps"]] == [
        "ok",
        "ok",
        "ok",
        "ok",
    ]


def test_agent_persists_derived_graph_once(monkeypatch, tmp_path):
    saved = []
    monkeypatch.setattr(
        "agentoi.agents.core.agent.save_run_graph",
        lambda **payload: saved.append(payload),
    )
    agent = _agent(
        tmp_path,
        save_traces=True,
        dynamo_table="agentoi-runs",
        save_local=False,
    )

    agent.invoke("Persist ontology trace.")

    assert len(saved) == 1
    assert saved[0]["table_name"] == "agentoi-runs"
    assert saved[0]["memory_trace"]["id"] == saved[0]["run_id"]
    assert saved[0]["graph"]["thoughts"]


def test_duckdb_store_is_written_before_dynamodb(monkeypatch, tmp_path):
    saved = []
    monkeypatch.setenv("AGENTOI_TRACE_STORE", "duckdb")
    monkeypatch.setenv("AGENTOI_TRACE_DB", str(tmp_path / "agentoi.duckdb"))
    monkeypatch.setattr(
        "agentoi.agents.core.agent.save_run_graph",
        lambda **payload: saved.append(payload),
    )
    agent = _agent(tmp_path, save_traces=True, dynamo_table="agentoi-runs", save_local=True)

    agent.invoke("Persist ontology trace.")

    assert saved == []
    from agentoi.agents.runtime.duckdb_store import list_runs

    assert len(list_runs(tmp_path / "agentoi.duckdb")) == 1


def test_load_run_graph_accepts_legacy_flattened_rows(monkeypatch):
    class Table:
        def query(self, **_kwargs):
            return {
                "Items": [
                    {"PK": "RUN#run-1", "SK": "META", "ItemType": "RUN"},
                    {
                        "PK": "RUN#run-1",
                        "SK": "NODE#n1",
                        "ItemType": "NODE",
                        "NodeId": "n1",
                        "Type": "prompt",
                        "Content": "question",
                    },
                    {
                        "PK": "RUN#run-1",
                        "SK": "EDGE#e1",
                        "ItemType": "EDGE",
                        "EdgeId": "e1",
                        "Src": "n1",
                        "Dst": "n2",
                        "Kind": "sequence",
                    },
                ]
            }

    monkeypatch.setattr(persistence, "_table", lambda _name: Table())
    monkeypatch.setattr(persistence, "_key_schema", lambda _name: ("PK", "SK"))

    loaded = persistence.load_run_graph(table_name="runs", run_id="run-1")

    assert loaded["graph"]["thoughts"][0]["content"] == "question"
    assert loaded["graph"]["edges"][0]["src"] == "n1"
