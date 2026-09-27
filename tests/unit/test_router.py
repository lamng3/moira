"""Understanding router: replay a paraphrase, otherwise retrieve."""

from agentoi.routing.understand import ActionRoute, understand_question
from agentoi.workspace import OntologyWorkspace


ONTOLOGY = """
@prefix ex: <https://example.test/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

ex:ontology a owl:Ontology .
ex:Parent a owl:Class ; rdfs:label "Parent" .
ex:Child a owl:Class ;
    rdfs:subClassOf ex:Parent ;
    rdfs:label "Child" .
"""

CACHED = "what organ systems are part of the mouse"
PARAPHRASE = "list the organ systems in the mouse"


class _Reply:
    def __init__(self, content: str) -> None:
        self.content = content


class _LLM:
    def __init__(self, content: str | None = None, error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.calls = 0

    def invoke(self, _prompt: str):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return _Reply(self.content or "")


def test_replay_names_a_cached_question() -> None:
    route = understand_question(
        PARAPHRASE,
        [CACHED],
        _LLM(
            '{"same_intent":{"noul":0.91},"match":{"choice":"q0"},'
            '"closeness":{"score":"Same question"}}'
        ),
    )

    assert route == ActionRoute(
        "replay",
        question=CACHED,
        same_intent=0.91,
        closeness="Same question",
    )
    assert route.note() == (
        f"Routed to a remembered question: {CACHED}. "
        "(same intent 0.91, closeness Same question)."
    )


def test_unknown_label_and_failed_call_fall_open() -> None:
    missing = understand_question(PARAPHRASE, [CACHED], _LLM("not json"))
    unknown = understand_question(
        PARAPHRASE,
        [CACHED],
        _LLM('{"match":{"choice":"something else"}}'),
    )
    failed = understand_question(PARAPHRASE, [CACHED], _LLM(error=RuntimeError("down")))

    retrieve = understand_question(
        PARAPHRASE,
        [CACHED],
        _LLM('{"match":{"choice":"retrieve"},"closeness":{"score":"Different"}}'),
    )

    assert missing.action == "retrieve"
    assert unknown.action == "retrieve"
    assert retrieve.action == "retrieve"
    assert retrieve.closeness == "Different"
    assert failed.action == "retrieve"
    assert failed.fail_open is True


def test_harness_choice_replays_without_the_answer_model(tmp_path, monkeypatch) -> None:
    ontology = tmp_path / "example.ttl"
    ontology.write_text(ONTOLOGY, encoding="utf-8")
    workspace = OntologyWorkspace(ontology)
    workspace._query_memory().hot.put(CACHED, "The mouse has a circulatory system.")

    class Router:
        def invoke(self, _prompt: str) -> _Reply:
            return _Reply(
                '{"same_intent":{"noul":0.8},"match":{"choice":"q0"},'
                '"closeness":{"score":"Same question"}}'
            )

    monkeypatch.setattr(OntologyWorkspace, "_router_llm", lambda self: Router())

    def answer_model(*_args, **_kwargs):
        raise AssertionError("the answer model should stay unused")

    monkeypatch.setattr("agentoi.workspace.create_application_agent", answer_model)

    answer = workspace.ask(PARAPHRASE)

    assert answer == "The mouse has a circulatory system."
    assert workspace.last_answer_source == "route"
    assert workspace.last_thought_seconds is not None
    assert workspace.last_thought_seconds >= 0
    assert "same intent 0.80" in (workspace.last_route_note or "")
    assert "closeness Same question" in (workspace.last_route_note or "")


def test_paraphrase_replays_without_the_answer_model(tmp_path, monkeypatch) -> None:
    ontology = tmp_path / "example.ttl"
    ontology.write_text(ONTOLOGY, encoding="utf-8")
    workspace = OntologyWorkspace(ontology)
    workspace._query_memory().hot.put(CACHED, "The mouse has a circulatory system.")
    monkeypatch.setattr(OntologyWorkspace, "_router_llm", lambda self: object())

    def route(_question, cached, _llm):
        return ActionRoute("replay", question=cached[0])

    monkeypatch.setattr("agentoi.routing.understand.understand_question", route)

    def answer_model(*_args, **_kwargs):
        raise AssertionError("the answer model should stay unused")

    monkeypatch.setattr("agentoi.workspace.create_application_agent", answer_model)

    answer = workspace.ask(PARAPHRASE)

    assert answer == "The mouse has a circulatory system."
    assert workspace.last_answer_source == "route"
    assert workspace.last_thought_seconds is not None
    assert workspace.last_thought_seconds >= 0
    assert workspace.last_route_note == f"Routed to a remembered question: {CACHED}."


def _stub_answer_path(monkeypatch, answer: str) -> None:
    from agentoi.algorithms.graph import ConceptGraph

    def skip_embeddings(self, alpha=0.5, progress=None, control=None):
        return self

    class Agent:
        def invoke(self, _prompt):
            return answer

    monkeypatch.setattr(ConceptGraph, "compute_all_embeddings", skip_embeddings)
    monkeypatch.setattr(
        "agentoi.workspace.build_context_candidates",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        ConceptGraph,
        "make_prompt_for_query",
        lambda self, *_args, **_kwargs: "prompt",
    )
    monkeypatch.setattr("agentoi.workspace.create_model_runtime", lambda **_kwargs: object())
    monkeypatch.setattr(
        "agentoi.workspace.create_application_agent",
        lambda *_args, **_kwargs: Agent(),
    )


def test_retrieve_still_calls_the_answer_model(tmp_path, monkeypatch) -> None:
    ontology = tmp_path / "example.ttl"
    ontology.write_text(f"{ONTOLOGY}\n# {tmp_path}\n", encoding="utf-8")
    workspace = OntologyWorkspace(ontology)
    workspace._query_memory().hot.put(CACHED, "cached")
    monkeypatch.setattr(OntologyWorkspace, "_router_llm", lambda self: object())
    monkeypatch.setattr(
        "agentoi.routing.understand.understand_question",
        lambda *_args, **_kwargs: ActionRoute("retrieve"),
    )
    _stub_answer_path(monkeypatch, "From the model.")

    answer = workspace.ask(PARAPHRASE)

    assert answer == "From the model."
    assert workspace.last_answer_source == "model"


def test_failed_router_falls_open_to_the_answer_model(tmp_path, monkeypatch) -> None:
    ontology = tmp_path / "example.ttl"
    ontology.write_text(f"{ONTOLOGY}\n# fail {tmp_path}\n", encoding="utf-8")
    workspace = OntologyWorkspace(ontology)
    workspace._query_memory().hot.put(CACHED, "cached")

    class Broken:
        def invoke(self, _prompt):
            raise RuntimeError("router down")

    monkeypatch.setattr(OntologyWorkspace, "_router_llm", lambda self: Broken())
    _stub_answer_path(monkeypatch, "Retrieved.")

    answer = workspace.ask(PARAPHRASE)

    assert answer == "Retrieved."
    assert workspace.last_answer_source == "model"


def test_exact_question_skips_the_router(tmp_path, monkeypatch) -> None:
    ontology = tmp_path / "example.ttl"
    ontology.write_text(ONTOLOGY, encoding="utf-8")
    workspace = OntologyWorkspace(ontology)
    workspace._query_memory().hot.put(CACHED, "cached answer")

    def router(*_args, **_kwargs):
        raise AssertionError("an exact hot hit should skip the router")

    monkeypatch.setattr("agentoi.routing.understand.understand_question", router)
    monkeypatch.setattr(OntologyWorkspace, "_router_llm", lambda self: object())

    answer = workspace.ask("What organ systems are part of the mouse?")

    assert answer == "cached answer"
    assert workspace.last_answer_source == "hot"
    assert workspace.last_route_note is None
