import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from moira.functions.similarity import (
    NormalizedGoogleDistance,
)
from moira.retrieval.web import (
    DatasetSearchProvider,
    GoogleCSEProvider,
    MediaWikiSearchProvider,
    StaticSearchProvider,
    VocabularyStats,
    WebSearchProvider,
    WebSearchPlugin,
    available_web_searches,
    create_web_search,
    describe_web_search,
    register_web_search,
    unregister_web_search,
)


def _node(node_id: str, label: str):
    concept = SimpleNamespace(
        name=label,
        ground_set={"labels": [label], "exact_synonyms": []},
    )
    return SimpleNamespace(id=node_id, equiv_concepts=[concept])


def _graph(*nodes):
    return SimpleNamespace(nodes={node.id: node for node in nodes})


def test_public_provider_types_and_default_are_offline():
    plugin = NormalizedGoogleDistance()

    assert isinstance(plugin.search_provider, StaticSearchProvider)
    assert issubclass(StaticSearchProvider, WebSearchProvider)
    assert VocabularyStats.__name__ == "VocabularyStats"
    assert available_web_searches() == (
        "dataset",
        "google-cse",
        "mediawiki",
        "static",
    )
    assert create_web_search("static").corpus_size() == 10**12
    assert describe_web_search("google-cse").environment_variables == (
        "GOOGLE_API_KEY",
        "GOOGLE_CSE_CX",
    )


def test_custom_web_search_provider_is_a_visible_plugin():
    register_web_search(
        WebSearchPlugin("project", "Project index", False),
        StaticSearchProvider,
    )
    try:
        assert "project" in available_web_searches()
        assert create_web_search("project", counts={"term": 7}).search_count(
            "term"
        ) == 7
    finally:
        unregister_web_search("project")


def test_ngd_uses_provider_corpus_size():
    provider = StaticSearchProvider(
        {"x": 100, "y": 50, "x y": 10},
        corpus_size=1000,
    )
    plugin = NormalizedGoogleDistance(provider)

    expected = (math.log(100) - math.log(10)) / (
        math.log(1000) - math.log(50)
    )
    assert plugin.compute_ngd("x", "y") == pytest.approx(expected)


def test_identical_terms_have_zero_distance():
    plugin = NormalizedGoogleDistance(StaticSearchProvider())

    assert plugin.compute_ngd("same_term", "same term") == 0.0
    assert plugin.compute_similarity("same_term", "same term") == 1.0


def test_ngd_edge_cases_fail_soft_and_stay_bounded():
    empty = StaticSearchProvider({}, default_count=0)
    assert NormalizedGoogleDistance(empty).compute_ngd("x", "y") == 1.0

    inconsistent = StaticSearchProvider(
        {"x": 10, "y": 10, "x y": 100},
        corpus_size=1000,
    )
    assert NormalizedGoogleDistance(inconsistent).compute_ngd("x", "y") == 0.0

    failing = Mock(spec=WebSearchProvider)
    failing.search_count.side_effect = RuntimeError("offline")
    failing.co_occurrence_count.side_effect = RuntimeError("offline")
    failing.corpus_size.side_effect = RuntimeError("offline")
    assert NormalizedGoogleDistance(failing).compute_similarity("x", "y") == 0.0


def test_unknown_terms_do_not_become_perfect_matches():
    plugin = NormalizedGoogleDistance(StaticSearchProvider())

    assert plugin.compute_similarity("unrelated alpha", "unrelated beta") == 0.0


def test_invalid_ids_are_filtered_from_all_results_and_top_pair_mapping_is_valid():
    graph = _graph(_node("a", "animal"), _node("b", "dog"), _node("c", "cat"))
    plugin = NormalizedGoogleDistance()
    requested = ["missing", "a", "c"]

    assert plugin.compute_pairwise_similarity(graph, requested).shape == (2, 2)
    assert plugin.compute_similarity_to_query(graph, "animal", requested).shape == (2,)
    pairs = plugin.get_top_similar_pairs(graph, k=10, node_ids=requested)
    nodes = plugin.get_most_similar_nodes(graph, "animal", k=10, node_ids=requested)
    assert pairs[0][:2] == ("a", "c")
    assert {node_id for node_id, _ in nodes} == {"a", "c"}
    assert plugin.representative_term(graph.nodes["a"]) == "animal"


def test_top_pair_indices_map_back_to_the_selected_upper_triangle_coordinates():
    graph = _graph(
        _node("a", "alpha"),
        _node("b", "beta"),
        _node("c", "gamma"),
        _node("d", "delta"),
    )
    provider = StaticSearchProvider(
        {
            "alpha": 100,
            "beta": 100,
            "gamma": 100,
            "delta": 100,
            "alpha beta": 1,
            "alpha gamma": 2,
            "alpha delta": 3,
            "beta gamma": 4,
            "beta delta": 90,
            "gamma delta": 100,
        },
        corpus_size=1000,
    )
    plugin = NormalizedGoogleDistance(provider)

    pairs = plugin.get_top_similar_pairs(graph, k=2)

    assert [(left, right) for left, right, _ in pairs] == [
        ("c", "d"),
        ("b", "d"),
    ]
    assert pairs[0][2] == pytest.approx(1.0)


def test_dataset_aware_counts_are_seeded_without_touching_global_random_state():
    graph = _graph(_node("a", "alpha"), _node("b", "beta"), _node("c", "gamma"))

    first = DatasetSearchProvider(
        graph, seed=7, co_occurrence_probability=1.0
    )
    second = DatasetSearchProvider(
        graph, seed=7, co_occurrence_probability=1.0
    )
    different = DatasetSearchProvider(
        graph, seed=8, co_occurrence_probability=1.0
    )

    assert first.counts == second.counts
    assert first.counts != different.counts
    assert first.get_vocabulary_stats().total_terms == 6
    assert first.get_vocabulary_stats().vocabulary_coverage == pytest.approx(0.5)


def test_google_retries_are_bounded_and_fail_soft():
    session = Mock()
    session.headers = {}
    session.get.side_effect = requests.ConnectionError("offline")
    provider = GoogleCSEProvider(
        "key",
        "cx",
        session=session,
        max_attempts=3,
        backoff=0,
    )

    assert provider.search_count("animal") == 0
    assert session.get.call_count == 3


def test_transient_provider_failure_is_not_cached():
    response = Mock(status_code=200)
    response.json.return_value = {
        "searchInformation": {"totalResults": "42"}
    }
    session = Mock()
    session.headers = {}
    session.get.side_effect = [requests.ConnectionError("offline"), response]
    provider = GoogleCSEProvider(
        "key", "cx", session=session, max_attempts=1, backoff=0
    )

    assert provider.search_count("animal") == 0
    assert provider.search_count("animal") == 42
    assert session.get.call_count == 2


def test_mediawiki_corpus_size_uses_api_and_is_cached():
    response = Mock(status_code=200)
    response.json.return_value = {
        "query": {"statistics": {"articles": 1234, "pages": 5678}}
    }
    session = Mock()
    session.headers = {}
    session.get.return_value = response
    provider = MediaWikiSearchProvider(session=session, backoff=0)

    assert provider.corpus_size() == 1234
    assert provider.corpus_size() == 1234
    assert session.get.call_count == 1
