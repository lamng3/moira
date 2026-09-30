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
SHORT_DEFINITION = (
    "An anatomic region is a part of the body with compartmental boundaries today."
)
SHORT_REVERSE = "What is a part of the body with compartmental boundaries today?"
SHORT_TYPO = SHORT_REVERSE.replace("compartmental", "compartemental")
SHORT_TWO = SHORT_TYPO.replace("boundaries", "bounderies")
TOOL_PLAN = (
    "To answer the user query precisely, I will use the tools provided. "
    '{"tool_name": "search_term_context", "arguments": {"term": ["mouse organ systems"]}}'
)


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


def test_one_typo_still_names_the_concept(tmp_path) -> None:
    memory = AgentMemory(MemoryConfig(directory=tmp_path))
    memory.remember(QueryPath.from_steps(ORIGINAL, [CONCEPT]), SHORT_DEFINITION)

    decision = memory.consult_answer(SHORT_TYPO)

    assert decision.source == "reverse"
    assert decision.answer == "Anatomic region"


def test_two_typos_do_not_match(tmp_path) -> None:
    memory = AgentMemory(MemoryConfig(directory=tmp_path))
    memory.remember(QueryPath.from_steps(ORIGINAL, [CONCEPT]), SHORT_DEFINITION)

    assert memory.consult_answer(SHORT_TWO).answer is None


def test_a_cached_tool_plan_does_not_replay(tmp_path, monkeypatch) -> None:
    ontology = tmp_path / "anatomy.ttl"
    ontology.write_text("@prefix ex: <https://example.test/> .\n", encoding="utf-8")
    monkeypatch.setenv("AGENTOI_CACHE_DIR", str(tmp_path / "cache"))
    workspace = OntologyWorkspace(ontology)
    memory = workspace._query_memory()
    memory.remember(QueryPath.from_steps(ORIGINAL, [CONCEPT]), SHORT_DEFINITION)
    memory.hot.put(SHORT_TYPO, TOOL_PLAN)

    def unused(*_args, **_kwargs):
        raise AssertionError("the answer model should stay unused")

    monkeypatch.setattr("agentoi.workspace.create_application_agent", unused)
    answer = workspace.ask(SHORT_TYPO)

    assert answer == "Anatomic region"
    assert workspace.last_answer_source == "reverse"


def test_a_tool_plan_is_not_shown_or_remembered(tmp_path, monkeypatch) -> None:
    from agentoi.algorithms.graph import ConceptGraph

    ontology = tmp_path / "anatomy.ttl"
    ontology.write_text(
        "@prefix ex: <https://example.test/> .\n"
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
        "ex:ontology a owl:Ontology .\n"
        'ex:Heart a owl:Class ; rdfs:label "Heart" .\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENTOI_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("AGENTOI_HARNESS_URL", "")
    monkeypatch.setattr(
        ConceptGraph,
        "compute_all_embeddings",
        lambda self, alpha=0.5, progress=None, control=None: self,
    )
    monkeypatch.setattr("agentoi.workspace.build_context_candidates", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        ConceptGraph,
        "make_prompt_for_query",
        lambda self, *_args, **_kwargs: "QUERY: Where is the spleen?",
    )
    monkeypatch.setattr("agentoi.workspace.create_model_runtime", lambda **_kwargs: object())

    class Agent:
        def invoke(self, _prompt: str) -> str:
            return TOOL_PLAN

    monkeypatch.setattr(
        "agentoi.workspace.create_application_agent",
        lambda *_args, **_kwargs: Agent(),
    )
    workspace = OntologyWorkspace(ontology)

    answer = workspace.ask("Where is the spleen?")

    assert answer == "The ontology did not yield an answer."
    assert workspace._query_memory().lookup_question("Where is the spleen?") is None


def test_a_cached_prompt_echo_keeps_the_concept_names(tmp_path, monkeypatch) -> None:
    ontology = tmp_path / "anatomy.ttl"
    ontology.write_text("@prefix ex: <https://example.test/> .\n", encoding="utf-8")
    monkeypatch.setenv("AGENTOI_CACHE_DIR", str(tmp_path / "cache"))
    workspace = OntologyWorkspace(ontology)
    question = "List the organ systems in the mouse"
    workspace._query_memory().hot.put(
        question,
        "The organ systems in the mouse include:\n\n"
        "* Visceral organ system\n\n"
        "These concepts answer the question by referencing specific terms from the "
        "RELEVANT CONCEPT CLUSTERS and KEY RELATIONS.",
    )

    def unused(*_args, **_kwargs):
        raise AssertionError("the answer model should stay unused")

    monkeypatch.setattr("agentoi.workspace.create_application_agent", unused)
    answer = workspace.ask(question)

    assert "Visceral organ system" in answer
    assert "RELEVANT CONCEPT CLUSTERS" not in answer
    assert workspace.last_answer_source == "hot"


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

    assert answer.startswith("An anatomic region is a part of the body")
    assert "RELEVANT CONCEPT CLUSTERS" not in answer
    assert workspace.last_answer_source == "hot"
