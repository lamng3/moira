from agentoi.agents import OOTAgentObserver
from agentoi.agents.core.events import emit_event
from agentoi.agents.ontology_of_thought.memory import ThoughtType as MemoryThoughtType
from agentoi.agents.ontology_of_thought.models import ThoughtType


def test_thought_type_is_shared_by_memory_and_graph_models():
    assert MemoryThoughtType is ThoughtType


def test_emit_event_delivers_once_when_observer_is_registered_twice():
    observer = OOTAgentObserver(save_traces=False)

    emit_event(
        "trace.start",
        observers=[observer, observer.on_event],
        oot_observer=observer,
        run_id="run-1",
        initial_prompt="prompt",
    )
    emit_event(
        "trace.end",
        observers=[observer],
        oot_observer=observer,
        run_id="run-1",
        result_preview="answer",
    )

    memory = observer.get_latest_memory_trace()
    assert memory is not None
    assert len(observer.completed_traces) == 1
    assert len(memory.nodes) == 2


def test_observer_memory_is_source_for_derived_graph():
    observer = OOTAgentObserver(save_traces=False)
    observer({"event": "trace.start", "run_id": "run-2", "initial_prompt": "prompt"})
    observer({"event": "llm.response", "preview": "answer"})
    observer({"event": "trace.end", "run_id": "run-2", "result_preview": "final"})

    memory = observer.get_latest_memory_trace()
    graph = observer.get_latest_thought_graph()

    assert memory is not None
    assert graph is not None
    assert set(graph.thoughts) == set(memory.nodes)
    run_path = graph.get_run_path("run-2")
    assert run_path is not None
    assert run_path.start == memory.root_node_id
    assert run_path.end == memory.current_node_id
