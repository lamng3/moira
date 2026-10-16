"""MOIRA on a shared Mneme memory. Skips unless mneme is installed or cloned beside moira."""

import os
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[3]
MNEME = SOURCE / "mneme" / "src"
if MNEME.is_dir() and str(MNEME) not in sys.path:
    sys.path.append(str(MNEME))
mneme = pytest.importorskip("mneme")

from moira.integrations.mneme import (  # noqa: E402
    MnemeTraceObserver,
    MoiraAnswerCache,
    OntoDBEngine,
    import_trace,
)

ONTODB = next(
    (p for p in [os.environ.get("ONTODB_BIN", ""), str(SOURCE / "ontodb/build/dev/ontodb")] if p and Path(p).is_file()),
    None,
)


def test_the_cache_serves_only_live_answers():
    store = mneme.MemoryStore()
    cache = MoiraAnswerCache()
    sync = mneme.CacheSync(store, cache, kinds=["answer"])
    version = store.put("aegis", "version", "v1")
    body = {"question": "What kinds of material are there?", "answer": "Metal, Polymer", "concepts": ["Material"]}
    store.put("moira", "answer", body["question"], body, derived_from=[version.id])
    assert cache.lookup("what kinds of material are there") == "Metal, Polymer"  # MOIRA's normalization
    store.invalidate([version.id], "rolled back")
    assert cache.lookup("What kinds of material are there?") is None
    assert sync.evicted == 1 and cache.rebuilds == 1
    cache.close()


def test_agent_events_become_a_trace_as_they_happen():
    store = mneme.MemoryStore()
    observer = MnemeTraceObserver(store)
    events = [
        {"event": "trace.start", "run_id": "run-1", "initial_prompt": "What is a Metal?"},
        {"event": "llm.request", "model": "m", "message_preview": "..."},
        {"event": "llm.response", "preview": "I should look it up"},
        {"event": "tool.parsed", "tool_call": {"tool_name": "ontology_term_info", "arguments": {"term": "Metal"}}},
        {"event": "tool.result", "tool_name": "ontology_term_info", "preview": "Metal subClassOf Material"},
        {"event": "trace.end", "result_preview": "A Metal is a Material."},
    ]
    for evt in events:
        observer(evt)
    (t,) = observer.completed
    back = mneme.Trace.open(store, t.id)
    assert [m.kind for m in back.steps] == ["llm_request", "thought", "action", "observation", "result", "trace_end"]
    action, seen = back.steps[2], back.steps[3]
    assert seen.body["parent"] == action.id and action.body["tool"] == "ontology_term_info"
    assert back.status() == "ok" and back.root.body["run"] == "run-1"


def test_a_saved_memory_trace_imports_with_its_tree():
    nodes = {
        "a": {"type": "prompt", "content": "Persist ontology trace.", "children_ids": ["b"], "metadata": {}},
        "b": {"type": "reasoning", "content": "request", "parent_id": "a", "children_ids": ["c"], "metadata": {}},
        "c": {"type": "tool_call", "content": "echo", "parent_id": "b", "children_ids": ["d"], "metadata": {}},
        "d": {"type": "tool_result", "content": "ok", "parent_id": "c", "children_ids": [], "metadata": {}},
    }
    store = mneme.MemoryStore()
    t = import_trace(store, {"id": "run-2", "root_node_id": "a", "nodes": nodes})
    back = mneme.Trace.open(store, t.id)
    assert back.root.body["task"] == "Persist ontology trace."
    assert [m.kind for m in back.steps] == ["llm_request", "action", "observation", "trace_end"]
    assert back.steps[2].body["parent"] == back.steps[1].id


@pytest.mark.skipif(ONTODB is None, reason="no ontodb binary")
def test_ontodb_holds_the_memory_log_across_processes(tmp_path):
    store = mneme.MemoryStore(OntoDBEngine(ONTODB, tmp_path / "db"))
    version = store.put("aegis", "version", "v1", {"note": 'quotes " and } braces\nand é'}, tier=mneme.LONG)
    answer = store.put("moira", "answer", "q", {"answer": "Metal, Polymer"}, derived_from=[version.id, "file:x y"])
    store.invalidate([version.id], "rolled back")

    again = mneme.MemoryStore(OntoDBEngine(ONTODB, tmp_path / "db"))  # a fresh OntoDB process reads it back
    assert again.state.status == {version.id: mneme.STALE, answer.id: mneme.STALE}
    assert again.tier(version.id) == mneme.LONG
    assert again.get(version.id).body["note"] == 'quotes " and } braces\nand é'
    assert again.engine.derived_from(version.id) == [answer.id]
    assert again.engine.shell.verify().endswith("verify: ok")
