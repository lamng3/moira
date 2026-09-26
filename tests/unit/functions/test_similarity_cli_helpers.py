from types import SimpleNamespace

import pytest

from agentoi.functions.similarity import (
    SimilarityMethod,
    analyze_alignment_similarity,
)


class _SimilaritySystem:
    def compute_similarity(
        self, graph, source_id, target_id, method, return_all=False
    ):
        assert method is SimilarityMethod.HYBRID
        assert return_all is True
        return {
            ("s1", "t1"): (0.9, 0.1, 0.2),
            ("s1", "t2"): (0.8, 0.1, 0.2),
            ("s2", "t2"): (0.2, 0.9, 0.2),
        }[(source_id, target_id)]


def test_similarity_helper_uses_positive_gold_and_per_source_candidates():
    nodes = {
        node_id: SimpleNamespace(id=node_id, equiv_concepts=[])
        for node_id in ("s1", "s2", "t1", "t2")
    }
    graph = SimpleNamespace(nodes=nodes)

    result = analyze_alignment_similarity(
        graph,
        _SimilaritySystem(),
        {("s1", "t1"), ("s2", "t2")},
        similarity_mode=SimilarityMethod.COSINE,
        similarity_threshold=0.5,
        gold_by_source=None,
        top_ids_by_source={"s1": ["t1", "t2"], "s2": ["t2"]},
    )

    assert result["similarity_avgs_over_gold"]["cosine"] == pytest.approx(0.55)
    assert result["threshold_metrics"] == {
        "threshold": 0.5,
        "precision": pytest.approx(0.5),
        "recall": pytest.approx(0.5),
        "f1": pytest.approx(0.5),
        "TP": 1,
        "|predicted|": 2,
    }
