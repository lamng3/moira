from types import SimpleNamespace

import pytest

import moira.uq as uq
from moira.uq import AgreementConfidenceScorer
from moira.uq.ensemble_scorer import EnsembleConfidenceScorer
from moira.uq.graph_scorer import GraphConfidenceScorer
from moira.uq.scorer import UQLMConfidenceScorer


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)

    def invoke(self, prompt):
        return SimpleNamespace(content=next(self.responses))


class FakeBackend:
    def score(self, **kwargs):
        assert kwargs["responses"] == ["YES"]
        assert kwargs["sampled_responses"] == [["YES", "YES", "NO"]]
        return SimpleNamespace(data={"noncontradiction": [1.2]})


class FakeScorer:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def score_prompt_from_responses(self, responses):
        return True, self.value, {self.name: self.value}


class FakeNLI:
    def predict(self, left, right):
        return [[0.0, 0.0, 0.8]]


def test_package_import_exposes_dependency_free_scorer():
    assert uq.AgreementConfidenceScorer is AgreementConfidenceScorer


def test_yes_no_parser_avoids_substrings_and_ambiguity():
    assert uq.parse_yes_no("YES, they match.") is True
    assert uq.parse_yes_no('<Answer>no</Answer>') is False
    assert uq.parse_yes_no("yesterday") is None
    assert uq.parse_yes_no("YES or NO") is None


def test_agreement_uses_winning_fraction_for_no_majority():
    answer, confidence, scores = AgreementConfidenceScorer().score_prompt_from_responses(
        ["NO", "No.", "YES"]
    )
    assert answer is False
    assert confidence == pytest.approx(2 / 3)
    assert scores == {"exact_match": pytest.approx(2 / 3)}


def test_agreement_generates_responses_through_shared_public_path():
    scorer = AgreementConfidenceScorer(FakeLLM(["YES", "NO", "YES"]), 3)
    assert scorer.score_prompt("compare")[0:2] == (True, pytest.approx(2 / 3))


def test_uqlm_backend_can_be_faked_and_clamps_scores():
    scorer = UQLMConfidenceScorer(
        llm=None,
        scorers=["noncontradiction"],
        primary_scorer="noncontradiction",
        backend=FakeBackend(),
    )
    answer, confidence, scores = scorer.score_prompt_from_responses(
        ["YES", "YES", "NO"]
    )
    assert answer is True
    assert confidence == 1.0
    assert scores["noncontradiction"] == 1.0


def test_graph_backend_uses_injected_nli_without_loading_a_model():
    scorer = GraphConfidenceScorer(
        llm=None,
        scorers=["degree_centrality"],
        primary_scorer="degree_centrality",
        nli=FakeNLI(),
    )
    answer, confidence, scores = scorer.score_prompt_from_responses(["YES", "YES"])
    assert answer is True
    assert confidence == pytest.approx(0.8)
    assert scores == {"degree_centrality": pytest.approx(0.8)}


def test_ensemble_uses_public_scorer_protocol_and_clamps_component_scores():
    scorer = EnsembleConfidenceScorer(
        scorers=[FakeScorer("first", 1.4), FakeScorer("second", -0.2)],
        weights={"first": 1, "second": 3},
    )
    answer, confidence, scores = scorer.score_prompt_from_responses(["YES"])
    assert answer is True
    assert confidence == pytest.approx(0.25)
    assert scores == {"first": 1.0, "second": 0.0, "ensemble": 0.25}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_responses": 0},
        {"weights": {}},
        {"weights": {"exact_match": -1}},
        {"weights": {"exact_match": 0}},
    ],
)
def test_ensemble_validates_inputs(kwargs):
    with pytest.raises((TypeError, ValueError)):
        EnsembleConfidenceScorer(**kwargs)


def test_scorers_validate_names_and_empty_responses():
    with pytest.raises(ValueError, match="unknown UQLM"):
        UQLMConfidenceScorer(None, scorers=["made_up"], backend=FakeBackend())
    with pytest.raises(ValueError, match="unknown graph"):
        GraphConfidenceScorer(None, scorers=["made_up"], nli=FakeNLI())
    with pytest.raises(ValueError, match="at least one response"):
        AgreementConfidenceScorer().score_prompt_from_responses([])
