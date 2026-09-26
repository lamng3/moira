"""Optional graph/NLI uncertainty backend."""

from __future__ import annotations

from typing import Any, Sequence

from agentoi.uq.common import (
    ScoreOutput,
    clamp_score,
    generate_responses,
    majority_result,
)

GRAPH_SCORERS = (
    "degree_centrality",
    "closeness_centrality",
    "betweenness_centrality",
    "page_rank",
    "harmonic_centrality",
    "laplacian_centrality",
)


class GraphConfidenceScorer:
    """Score semantic consistency using an optional NLI entailment graph."""

    def __init__(
        self,
        llm: Any,
        num_responses: int = 5,
        scorers: Sequence[str] | None = None,
        primary_scorer: str = "closeness_centrality",
        device: str | None = None,
        nli_model_name: str = "microsoft/deberta-large-mnli",
        *,
        nli: Any | None = None,
        networkx_module: Any | None = None,
    ) -> None:
        if isinstance(num_responses, bool) or not isinstance(num_responses, int):
            raise TypeError("num_responses must be an integer")
        if num_responses < 1:
            raise ValueError("num_responses must be positive")
        scorer_names = tuple(GRAPH_SCORERS if scorers is None else scorers)
        if not scorer_names or len(set(scorer_names)) != len(scorer_names):
            raise ValueError("scorers must be a non-empty list of unique names")
        unknown = set(scorer_names) - set(GRAPH_SCORERS)
        if unknown:
            raise ValueError(f"unknown graph scorer(s): {', '.join(sorted(unknown))}")
        if primary_scorer not in scorer_names:
            raise ValueError("primary_scorer must be included in scorers")

        if networkx_module is None:
            try:
                import networkx as networkx_module
            except ImportError as exc:
                raise ImportError(
                    "Graph scoring requires networkx; install it with "
                    "`pip install networkx`."
                ) from exc
        if nli is None:
            try:
                from uqlm.nli import NLI
            except ImportError as exc:
                raise ImportError(
                    "Graph scoring requires UQLM's NLI backend; install it with "
                    "`pip install 'agentoi[uq]'`."
                ) from exc
            nli = NLI(device=device, nli_model_name=nli_model_name)

        self.llm = llm
        self.num_responses = num_responses
        self._scorer_names = scorer_names
        self._primary_scorer = primary_scorer
        self._nli = nli
        self._nx = networkx_module

    def score_prompt(self, prompt: str) -> ScoreOutput:
        return self.score_prompt_from_responses(
            generate_responses(self.llm, prompt, self.num_responses)
        )

    def score_prompt_from_responses(
        self, raw_responses: Sequence[str]
    ) -> ScoreOutput:
        answer, agreement = majority_result(raw_responses)
        if len(raw_responses) < 2:
            scores = {name: agreement for name in self._scorer_names}
            return answer, scores[self._primary_scorer], scores
        scores = self._centrality(self._entailments(raw_responses))
        confidence = scores.get(self._primary_scorer, agreement)
        return answer, confidence, scores

    score_from_responses = score_prompt_from_responses

    def _entailments(self, responses: Sequence[str]) -> list[list[float]]:
        size = len(responses)
        matrix = [[0.0] * size for _ in range(size)]
        for left in range(size):
            for right in range(size):
                if left == right:
                    matrix[left][right] = 1.0
                else:
                    probabilities = self._nli.predict(
                        responses[left], responses[right]
                    )
                    matrix[left][right] = clamp_score(
                        probabilities[0][2], name="entailment"
                    )
        return matrix

    def _centrality(self, matrix: Sequence[Sequence[float]]) -> dict[str, float]:
        nx = self._nx
        size = len(matrix)
        graph = nx.Graph()
        graph.add_nodes_from(range(size))
        for left in range(size):
            for right in range(left + 1, size):
                weight = (matrix[left][right] + matrix[right][left]) / 2.0
                if weight > 0.01:
                    graph.add_edge(left, right, weight=weight)

        scores: dict[str, float] = {}
        if "degree_centrality" in self._scorer_names:
            degrees = dict(graph.degree(weight="weight"))
            scores["degree_centrality"] = _mean(
                degrees.get(node, 0.0) / max(size - 1, 1) for node in range(size)
            )
        if "closeness_centrality" in self._scorer_names:
            scores["closeness_centrality"] = _mean(
                nx.closeness_centrality(graph).values()
            )
        if "betweenness_centrality" in self._scorer_names:
            scores["betweenness_centrality"] = _mean(
                nx.betweenness_centrality(graph, weight=None).values()
            )
        if "page_rank" in self._scorer_names:
            try:
                scores["page_rank"] = _mean(
                    nx.pagerank(graph, weight="weight", max_iter=1000).values()
                )
            except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
                scores["page_rank"] = 0.0
        if "harmonic_centrality" in self._scorer_names:
            harmonic = nx.harmonic_centrality(graph)
            scores["harmonic_centrality"] = _mean(
                harmonic.get(node, 0.0) / max(size - 1, 1)
                for node in range(size)
            )
        if "laplacian_centrality" in self._scorer_names:
            try:
                scores["laplacian_centrality"] = _mean(
                    nx.laplacian_centrality(graph, weight="weight").values()
                )
            except (AttributeError, nx.NetworkXError):
                scores["laplacian_centrality"] = 0.0
        return {name: clamp_score(value, name=name) for name, value in scores.items()}


def _mean(values: Any) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0
