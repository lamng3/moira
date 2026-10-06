from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

from moira.agents.core.tool_executor import FALLBACK_MODULES, load_tools_json
from moira.agents.tools.common import coerce_terms
from moira.agents.tools.common.http import SessionHttp
from moira.agents.tools.common.resources import read_json_resource
from moira.agents.tools.registry import (
    default_registry_resource,
    load_registry,
)


def test_shared_term_coercion_handles_common_delimiters() -> None:
    expected = ["alpha", "beta", "gamma"]
    assert coerce_terms('["alpha", "beta", "gamma"]') == expected
    assert coerce_terms("alpha | beta; gamma") == expected


def test_shared_http_uses_session_defaults_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        text = "body"

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, bool]:
            return {"ok": True}

    class Session:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}
            self.calls: list[tuple[str, dict[str, object]]] = []
            self.closed = False

        def get(self, url: str, **kwargs):
            self.calls.append((url, kwargs))
            return Response()

        def close(self) -> None:
            self.closed = True

    session = Session()
    monkeypatch.setattr(
        "moira.agents.tools.common.http.requests.Session",
        lambda: session,
    )

    client = SessionHttp(default_timeout=9, headers={"Accept": "x/test"})
    assert client.get_json("https://example.test") == {"ok": True}
    assert session.headers == {"Accept": "x/test"}
    assert session.calls[0][1]["timeout"] == 9
    client.close()
    assert session.closed


def test_packaged_registry_loads_preserved_module_paths() -> None:
    resource = default_registry_resource()
    registry = load_registry()

    assert resource.is_file()
    assert registry["search_term_context"]["module_path"] == (
        "moira.agents.tools.website.website_lookup"
    )
    assert "search_term_context" not in FALLBACK_MODULES
    assert load_tools_json(resource) == registry


def test_registry_loader_keeps_lenient_custom_json_support(
    tmp_path: Path,
) -> None:
    registry_file = tmp_path / "tools.json"
    registry_file.write_text(
        """
        {
          // legacy registries may contain comments
          "lookup": {
            "tool_name": "lookup",
            "module_path": " example/tools.lookup ",
          },
        }
        """,
        encoding="utf-8",
    )

    assert load_registry(registry_file)["lookup"]["module_path"] == (
        "example.tools.lookup"
    )


def test_cypher_catalog_is_package_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    catalog = read_json_resource(
        "moira.agents.tools.knowledge_graph",
        "cyphers.json",
    )
    assert catalog["default"]["random_nodes"].startswith("MATCH (n)")


def test_tool_imports_do_not_load_dotenv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dotenv

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        dotenv,
        "load_dotenv",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    modules = (
        "moira.agents.tools.website.config",
        "moira.agents.tools.ontology.config",
        "moira.agents.tools.knowledge_graph.config",
    )
    for module_name in modules:
        sys.modules.pop(module_name, None)
        importlib.import_module(module_name)

    assert calls == []


def test_ontology_testbed_annotation_resolves() -> None:
    module = importlib.import_module(
        "moira.agents.tools.ontology.ontology_lookup"
    )
    assert module.TestbedName is str


def test_knowledge_graph_provider_failures_remain_fail_soft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modules_before = set(sys.modules)
    if importlib.util.find_spec("neo4j") is None:
        neo4j = types.ModuleType("neo4j")
        neo4j.Driver = object
        neo4j.GraphDatabase = types.SimpleNamespace(
            driver=lambda *args, **kwargs: None
        )
        monkeypatch.setitem(sys.modules, "neo4j", neo4j)
    module = importlib.import_module(
        "moira.agents.tools.knowledge_graph.knowledge_graph_lookup"
    )
    monkeypatch.setattr(module, "DEBUG", True)

    class RaisingPool:
        enabled_names: list[str] = []

        def search_nodes(self, *args, **kwargs):
            raise RuntimeError("pool unavailable")

    class RaisingNeo4j:
        enabled = True

        def search_nodes(self, *args, **kwargs):
            raise RuntimeError("neo4j unavailable")

    class RaisingPublic:
        def search(self, *args, **kwargs):
            raise RuntimeError("public provider unavailable")

    lookup = module.KnowledgeGraphLookup.__new__(module.KnowledgeGraphLookup)
    lookup.pool = RaisingPool()
    lookup.neo4j = RaisingNeo4j()
    lookup.dbpedia = RaisingPublic()
    lookup.wikidata = RaisingPublic()

    result = lookup.entity_search("heart", include_public=True)
    assert result["heart"]["items"] == []
    assert result["heart"]["count"] == 0

    for module_name in set(sys.modules) - modules_before:
        if module_name.startswith("moira.agents.tools.knowledge_graph"):
            sys.modules.pop(module_name, None)
