import json
import sys
import types

from agentoi.agents import Agent, JevModelRouter, OOTAgentObserver, ToolRiskGate
from agentoi.agents.decisions import create_model_router, create_tool_gate


class FakeLLM:
    def __init__(self, model="fake-model", payload=None):
        self.model = model
        self.payload = payload

    def invoke(self, _messages):
        call = self.payload or {
            "tool_name": "echo",
            "tool_type": "module",
            "arguments": {"text": "ontology"},
        }
        return types.SimpleNamespace(content=json.dumps(call))


def _agent(tmp_path, llm, **kwargs):
    module = types.ModuleType("agentoi_test_tools")
    called = []

    def echo(text):
        called.append(("echo", text))
        return {"text": text}

    def search_term_context(term, api_key=None):
        called.append(("search_term_context", term, api_key))
        return {"term": term}

    def ontology_term_info(term):
        called.append(("ontology_term_info", term))
        return {"term": term}

    module.echo = echo
    module.search_term_context = search_term_context
    module.ontology_term_info = ontology_term_info
    sys.modules[module.__name__] = module
    registry = tmp_path / "tools.json"
    registry.write_text(
        json.dumps(
            {
                name: {
                    "tool_name": name,
                    "tool_type": "module",
                    "module_path": module.__name__,
                }
                for name in ("echo", "search_term_context", "ontology_term_info")
            }
        ),
        encoding="utf-8",
    )
    agent = Agent(
        llm,
        tools_path=registry,
        oot_observer=OOTAgentObserver(save_traces=False),
        verbose=False,
        save_traces=False,
        **kwargs,
    )
    return agent, called


def _step(response, name):
    return next(step["data"] for step in response["trace"]["steps"] if step["name"] == name)


def _router(choice, factory, routes):
    def classify(_state, _questions):
        return {
            "choice": choice,
            "probabilities": {choice: 0.91, "standard": 0.09},
            "confidence": 0.8,
        }

    return JevModelRouter(
        classifier=classify,
        model_factory=factory,
        routes=routes,
    )


def test_router_selects_local_and_careful_models(tmp_path):
    built = []

    def factory(name):
        built.append(name)
        return FakeLLM(model=name)

    routes = {
        "local": "ollama:phi3",
        "standard": "fake-model",
        "careful": "careful-model",
    }
    local_agent, _called = _agent(
        tmp_path,
        FakeLLM(),
        model_router=_router("local", factory, routes),
    )
    local = local_agent.invoke("List the label.", return_trace=True)
    careful_agent, _called = _agent(
        tmp_path,
        FakeLLM(),
        model_router=_router("careful", factory, routes),
    )
    careful = careful_agent.invoke(
        "Resolve the alignment conflict across both ontologies.",
        return_trace=True,
    )

    local_route = _step(local, "model_route")
    careful_route = _step(careful, "model_route")
    assert built == ["ollama:phi3", "careful-model"]
    assert local_route["name"] == "local"
    assert local_route["confidence"] == 0.8
    assert _step(local, "llm_invoke")["route"]["model_name"] == "ollama:phi3"
    assert careful_route["probabilities"]["careful"] == 0.91
    assert "Routed to careful model careful-model." in json.dumps(careful["memory_trace"])
    assert local_agent.llm.model == "fake-model"
    assert careful_agent.llm.model == "fake-model"


def test_routing_fails_open_to_the_current_model(tmp_path):
    def classify(_state, _questions):
        raise RuntimeError("jev unavailable")

    def factory(name):
        raise AssertionError(name)

    agent, called = _agent(
        tmp_path,
        FakeLLM(),
        model_router=JevModelRouter(classifier=classify, model_factory=factory),
    )

    response = agent.invoke("What is freshwater?", return_trace=True)

    assert called == [("echo", "ontology")]
    route = _step(response, "model_route")
    assert route["fail_open"] is True
    assert route["model_name"] == "fake-model"
    assert "jev unavailable" in route["error"]


def test_gate_blocks_web_lookup_and_redacts_arguments(tmp_path):
    seen = []

    def classify(state, _questions):
        seen.append(state)
        return {"noul": 0.92}

    payload = {
        "tool_name": "search_term_context",
        "arguments": {
            "term": "freshwater",
            "api_key": "sk-abcdefghijklmnopqrstuvwxyz",
        },
    }
    agent, called = _agent(
        tmp_path,
        FakeLLM(payload=payload),
        tool_gate=ToolRiskGate(classifier=classify, threshold=0.5),
    )

    response = agent.invoke("Look this up on the web.", return_trace=True)

    assert called == []
    assert response["result"]["blocked"] is True
    assert response["result"]["risk"] == 0.92
    step = response["trace"]["steps"][-1]
    assert step["name"] == "tool_execute"
    assert step["status"] == "blocked"
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in seen[0]
    assert "******" in seen[0]
    assert "Blocked search_term_context" in json.dumps(response["memory_trace"])


def test_ungated_lookup_runs_even_when_jev_would_block(tmp_path):
    def classify(_state, _questions):
        raise AssertionError("ungated tools are not sent to Jev")

    payload = {
        "tool_name": "ontology_term_info",
        "arguments": {"term": "freshwater"},
    }
    agent, called = _agent(
        tmp_path,
        FakeLLM(payload=payload),
        tool_gate=ToolRiskGate(classifier=classify),
    )

    response = agent.invoke("Define freshwater.", return_trace=True)

    assert called == [("ontology_term_info", "freshwater")]
    assert response["trace"]["steps"][-1]["status"] == "ok"


def test_enabled_gate_fails_closed_for_gated_tools(tmp_path):
    def classify(_state, _questions):
        raise RuntimeError("jev unavailable")

    payload = {
        "tool_name": "search_term_context",
        "arguments": {"term": "freshwater"},
    }
    agent, called = _agent(
        tmp_path,
        FakeLLM(payload=payload),
        tool_gate=ToolRiskGate(classifier=classify),
    )

    response = agent.invoke("Search the web.", return_trace=True)

    assert called == []
    assert response["result"]["fail_closed"] is True
    assert response["trace"]["steps"][-1]["status"] == "blocked"


def test_disabled_gate_preserves_tool_execution(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENTOI_TOOL_GATE", raising=False)
    monkeypatch.delenv("AGENTOI_MODEL_ROUTING", raising=False)
    payload = {
        "tool_name": "search_term_context",
        "arguments": {"term": "freshwater"},
    }
    agent, called = _agent(tmp_path, FakeLLM(payload=payload))

    response = agent.invoke("Search the web.", return_trace=True)

    assert called == [("search_term_context", "freshwater", None)]
    assert response["result"]["term"] == "freshwater"
    assert create_model_router() is None
    assert create_tool_gate() is None
