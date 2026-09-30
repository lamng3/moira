"""A definition asked back names the concept without the answer model."""

from agentoi.memory.config import MemoryConfig
from agentoi.memory.path import QueryPath
from agentoi.memory.service import AgentMemory
from agentoi.workspace import OntologyWorkspace


DEFINITION = (
    "An anatomic region is a part of the body that has a specific location and "
    "function, but does not have well-defined compartmental boundaries. "
    "This concept answers the question by referencing specific terms from the "
    "RELEVANT CONCEPT CLUSTERS, including anatomic region."
)
REVERSE = (
    "What is a part of the body that has a specific location and function, "
    "but does not have well-defined compartmental boundaries?"
)
ORIGINAL = "What is an anatomic region?"
CONCEPT = "http://example.org/Anatomic region"


def _memory(tmp_path) -> AgentMemory:
    memory = AgentMemory(MemoryConfig(directory=tmp_path))
    memory.remember(QueryPath.from_steps(ORIGINAL, [CONCEPT]), DEFINITION)
    return memory


def test_a_reversed_definition_names_the_concept(tmp_path) -> None:
    memory = _memory(tmp_path)

    decision = memory.consult_answer(REVERSE)
    reloaded = AgentMemory(MemoryConfig(directory=tmp_path)).consult_answer(REVERSE)

    assert decision.source == "reverse"
    assert decision.answer == "Anatomic region"
    assert reloaded.answer == "Anatomic region"


def test_a_weak_overlap_does_not_match(tmp_path) -> None:
    memory = _memory(tmp_path)

    assert memory.consult_answer("where is the liver").answer is None
    assert memory.consult_answer(
        "What is the specific location of the mouse heart in the chest today?"
    ).answer is None


def test_reverse_question_skips_the_answer_model(tmp_path, monkeypatch) -> None:
    ontology = tmp_path / "anatomy.ttl"
    ontology.write_text("@prefix ex: <https://example.test/> .\n", encoding="utf-8")
    monkeypatch.setenv("AGENTOI_CACHE_DIR", str(tmp_path / "cache"))
    workspace = OntologyWorkspace(ontology)
    workspace._query_memory().remember(QueryPath.from_steps(ORIGINAL, [CONCEPT]), DEFINITION)

    def unused(*_args, **_kwargs):
        raise AssertionError("the answer model should stay unused")

    monkeypatch.setattr("agentoi.workspace.create_application_agent", unused)
    answer = workspace.ask(REVERSE)

    assert answer == "Anatomic region"
    assert workspace.last_answer_source == "reverse"
    assert workspace.last_route_note == "Matched a remembered definition: Anatomic region."


def test_exact_question_still_hits_the_hot_cache(tmp_path, monkeypatch) -> None:
    ontology = tmp_path / "anatomy.ttl"
    ontology.write_text("@prefix ex: <https://example.test/> .\n", encoding="utf-8")
    monkeypatch.setenv("AGENTOI_CACHE_DIR", str(tmp_path / "cache"))
    workspace = OntologyWorkspace(ontology)
    workspace._query_memory().remember(QueryPath.from_steps(ORIGINAL, [CONCEPT]), DEFINITION)

    def unused(*_args, **_kwargs):
        raise AssertionError("the answer model should stay unused")

    monkeypatch.setattr("agentoi.workspace.create_application_agent", unused)
    answer = workspace.ask(ORIGINAL)

    assert answer == DEFINITION
    assert workspace.last_answer_source == "hot"
