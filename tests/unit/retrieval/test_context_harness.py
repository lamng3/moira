import pytest

from moira.algorithms.graph import (
    Concept,
    ConceptGraph,
    EquivalentClass,
    EquivalentClassRelation,
)
from moira.retrieval import (
    ContextCandidate,
    JevContextHarness,
    PassthroughHarness,
    build_context_candidates,
    create_context_harness,
)
from moira.workspace import _append_web_context


def _candidate(identifier: str, source: str = "concept") -> ContextCandidate:
    return ContextCandidate(
        id=identifier,
        text=identifier,
        source=source,
        score=0.2 if identifier == "noise" else 0.9,
    )


def _selector(query, candidates, **options):
    assert query == "freshwater"
    assert options["scoring_strategy"] == "contextual"
    assert "hierarchy" in options["relevance_guidance"]
    kept = [candidate for candidate in candidates if candidate.id != "noise"]
    shadow = options["shadow"]
    return {
        "status": "shadow" if shadow else "applied",
        "selected_ids": [candidate.id for candidate in kept],
        "documents": list(candidates) if shadow else kept,
        "decisions": [
            {
                "id": candidate.id,
                "probability": candidate.score,
                "selected": candidate.id != "noise",
                "returned": shadow or candidate.id != "noise",
                "reason": "retained" if candidate.id != "noise" else "below_threshold",
            }
            for candidate in candidates
        ],
        "usage": {"total_tokens": 24},
    }


def test_passthrough_retains_every_candidate():
    candidates = [_candidate("water"), _candidate("noise")]

    selection = PassthroughHarness().select("freshwater", candidates)

    assert selection.status == "passthrough"
    assert selection.metrics()["rejection_rate"] == 0.0
    assert [candidate.id for candidate in selection.documents] == ["water", "noise"]


def test_shadow_records_proposal_without_changing_context():
    candidates = [_candidate("water"), _candidate("noise"), _candidate("liquid", "neighborhood")]

    selection = JevContextHarness(selector=_selector, shadow=True).select(
        "freshwater", candidates
    )

    assert selection.status == "shadow"
    assert selection.proposed_ids == ("water", "liquid")
    assert [candidate.id for candidate in selection.documents] == [
        "water",
        "noise",
        "liquid",
    ]
    assert selection.metrics()["selected_count"] == 2
    assert selection.metrics()["returned_count"] == 3
    assert selection.usage_tokens == 24


def test_live_selection_returns_only_accepted_context():
    candidates = [_candidate("water"), _candidate("noise")]

    selection = JevContextHarness(selector=_selector, shadow=False).select(
        "freshwater", candidates
    )

    assert selection.status == "applied"
    assert [candidate.id for candidate in selection.documents] == ["water"]


def test_provider_failure_fails_open():
    def fail(query, candidates, **options):
        raise TimeoutError("provider unavailable")

    candidates = [_candidate("water")]
    selection = JevContextHarness(selector=fail).select("freshwater", candidates)

    assert selection.status == "bypassed"
    assert selection.metrics()["fail_open"] is True
    assert selection.documents == tuple(candidates)
    assert "provider unavailable" in selection.error


def test_missing_credentials_fail_open(monkeypatch):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    candidates = [_candidate("water")]

    selection = JevContextHarness().select("freshwater", candidates)

    assert selection.status == "bypassed"
    assert "JEV_API_KEY" in selection.error


def test_context_candidates_include_neighbors_and_web_once():
    parent = EquivalentClass([Concept("liquid")])
    child = EquivalentClass([Concept("water")])
    graph = ConceptGraph(
        [parent, child],
        [EquivalentClassRelation(parent, child, "no", score=1.0)],
    )
    graph.nearest_nodes = lambda terms, k=10: [(child, 0.8)]

    candidates = build_context_candidates(
        graph,
        "freshwater",
        include_neighborhood=True,
        web_passages=[
            {"id": "wiki", "text": "Fresh water is water with low salinity."},
            {"id": "wiki", "text": "duplicate"},
            {"snippet": "A topical geology page."},
        ],
    )

    assert [(candidate.id, candidate.source) for candidate in candidates] == [
        ("water", "concept"),
        ("liquid", "neighborhood"),
        ("wiki", "web"),
        ("web-3", "web"),
    ]


def test_selected_concepts_and_web_passages_shape_the_prompt():
    parent = EquivalentClass([Concept("liquid")])
    child = EquivalentClass([Concept("water")])
    noise = EquivalentClass([Concept("geology")])
    graph = ConceptGraph(
        [parent, child, noise],
        [EquivalentClassRelation(parent, child, "no", score=1.0)],
    )

    prompt = graph.make_prompt_for_query(
        "What is freshwater?",
        ["freshwater"],
        selected_ids=["water", "liquid", "wiki"],
    )
    selection = JevContextHarness(selector=_selector, shadow=False).select(
        "freshwater",
        [_candidate("water"), _candidate("wiki", "web")],
    )

    assert "water" in prompt
    assert "liquid" in prompt
    assert "geology" not in prompt
    rendered = _append_web_context(prompt, selection)
    assert "WEB CONTEXT:" in rendered
    assert "wiki" in rendered


def test_factory_defaults_to_shadow_mode(monkeypatch):
    monkeypatch.delenv("MOIRA_CONTEXT_MODE", raising=False)

    assert isinstance(create_context_harness(), PassthroughHarness)
    harness = create_context_harness("jev")
    assert isinstance(harness, JevContextHarness)
    assert harness.shadow is True

    monkeypatch.setenv("MOIRA_CONTEXT_MODE", "live")
    assert create_context_harness("jev").shadow is False
    with pytest.raises(ValueError, match="passthrough"):
        create_context_harness("unknown")
