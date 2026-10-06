"""HumanGS-style query optimization for OOT graphs."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import cosine_similarity

from moira.embeddings import TextEmbedding

from .graph import ThoughtGraph


@dataclass
class NodeProfile:
    node_id: str
    rank: int
    pset_size: int
    rset_size: int
    pset: Set[str]
    rset: Set[str]
    semantic_embedding: Optional[torch.Tensor] = None

    def partition_score(self, total_nodes: int) -> float:
        return self.rset_size * (total_nodes - self.rset_size)


@dataclass
class Cluster:
    cluster_id: int
    node_ids: List[str]
    representative_id: Optional[str] = None
    priority_score: float = 0.0


class QueryOptimizer:
    def __init__(
        self,
        graph: ThoughtGraph,
        text_embedder: Optional[TextEmbedding] = None,
        min_cluster_size: int = 2,
        max_clusters: Optional[int] = None,
    ):
        self.graph = graph
        self.text_embedder = text_embedder or TextEmbedding()
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters
        self._profiles: Dict[str, NodeProfile] = {}
        self._ranks: Optional[Dict[str, int]] = None

    def _walk(self, node_id: str, adjacency: dict, endpoint: str) -> Set[str]:
        if node_id not in self.graph.thoughts:
            return set()
        result: Set[str] = set()
        visited: Set[str] = set()
        queue = deque([node_id])
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            for edge_id in adjacency.get(current, []):
                edge = self.graph.edges.get(edge_id)
                related = getattr(edge, endpoint) if edge else None
                if related and related not in result:
                    result.add(related)
                    queue.append(related)
        return result

    def compute_pset(self, node_id: str) -> Set[str]:
        return self._walk(node_id, self.graph.in_adj, "src")

    def compute_rset(self, node_id: str) -> Set[str]:
        return self._walk(node_id, self.graph.out_adj, "dst")

    def compute_ranks(self) -> Dict[str, int]:
        if self._ranks is not None:
            return self._ranks
        children = defaultdict(list)
        in_degree = defaultdict(int)
        nodes = set(self.graph.thoughts)
        for edge in self.graph.edges.values():
            children[edge.src].append(edge.dst)
            in_degree[edge.dst] += 1
            nodes.update((edge.src, edge.dst))
        ranks: Dict[str, int] = {}
        queue = deque(node for node in nodes if in_degree[node] == 0)
        for node in queue:
            ranks[node] = 0
        while queue:
            current = queue.popleft()
            for child in children[current]:
                in_degree[child] -= 1
                ranks[child] = max(ranks.get(child, 0), ranks[current] + 1)
                if in_degree[child] == 0:
                    queue.append(child)
        self._ranks = {node: ranks.get(node, 0) for node in nodes}
        return self._ranks

    def compute_node_profile(self, node_id: str) -> NodeProfile:
        if node_id in self._profiles:
            return self._profiles[node_id]
        if node_id not in self.graph.thoughts:
            raise ValueError(f"Node {node_id} not in graph")
        pset, rset = self.compute_pset(node_id), self.compute_rset(node_id)
        thought = self.graph.thoughts[node_id]
        text = thought.question or thought.answer or thought.content or ""
        profile = NodeProfile(
            node_id=node_id,
            rank=self.compute_ranks().get(node_id, 0),
            pset_size=len(pset),
            rset_size=len(rset),
            pset=pset,
            rset=rset,
            semantic_embedding=self.text_embedder.to_embedding(text) if text else None,
        )
        self._profiles[node_id] = profile
        return profile

    def compute_all_profiles(self) -> Dict[str, NodeProfile]:
        for node_id in self.graph.thoughts:
            self.compute_node_profile(node_id)
        return self._profiles

    def cluster_nodes(
        self,
        profiles: Optional[Dict[str, NodeProfile]] = None,
        method: str = "agglomerative",
        n_clusters: Optional[int] = None,
    ) -> List[Cluster]:
        profiles = profiles or self.compute_all_profiles()
        if not profiles:
            return []
        max_rank = max(profile.rank for profile in profiles.values()) or 1
        max_pset = max(profile.pset_size for profile in profiles.values()) or 1
        max_rset = max(profile.rset_size for profile in profiles.values()) or 1
        features, node_ids = [], []
        for node_id, profile in profiles.items():
            if profile.semantic_embedding is None:
                continue
            embedding = profile.semantic_embedding.numpy()
            normalized = embedding / (np.linalg.norm(embedding) + 1e-9)
            features.append(np.concatenate((
                normalized,
                [profile.rank / max_rank, profile.pset_size / max_pset, profile.rset_size / max_rset],
            )))
            node_ids.append(node_id)
        if not features:
            return []
        n_clusters = n_clusters or max(1, len(node_ids) // self.min_cluster_size)
        if self.max_clusters:
            n_clusters = min(n_clusters, self.max_clusters)
        algorithm = (
            KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            if method == "kmeans"
            else AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        )
        grouped = defaultdict(list)
        for index, label in enumerate(algorithm.fit_predict(np.asarray(features))):
            grouped[int(label)].append(node_ids[index])
        return [
            Cluster(cluster_id=cluster_id, node_ids=ids)
            for cluster_id, ids in grouped.items()
            if len(ids) >= self.min_cluster_size
        ]

    def select_median_rank_representative(
        self, cluster: Cluster, profiles: Dict[str, NodeProfile]
    ) -> str:
        members = sorted(
            (profiles[node_id] for node_id in cluster.node_ids if node_id in profiles),
            key=lambda profile: profile.rank,
        )
        if not members:
            return cluster.node_ids[0]
        median = members[(len(members) - 1) // 2]
        candidates = [
            profile for profile in members if abs(profile.rank - median.rank) <= 1
        ]
        return max(
            candidates,
            key=lambda profile: profile.partition_score(len(self.graph.thoughts)),
        ).node_id

    def compute_cluster_priority(
        self, cluster: Cluster, profiles: Dict[str, NodeProfile]
    ) -> float:
        members = [profiles[node_id] for node_id in cluster.node_ids if node_id in profiles]
        if not members:
            return 0.0
        if len(members) > 1 and all(item.semantic_embedding is not None for item in members):
            embeddings = torch.stack([item.semantic_embedding for item in members]).numpy()
            diversity = 1.0 - cosine_similarity(embeddings).mean()
        else:
            diversity = 0.5
        ranks = [item.rank for item in members]
        max_rank = max(profile.rank for profile in profiles.values()) or 1
        spread = (max(ranks) - min(ranks)) / max_rank
        return len(members) * (1.0 + diversity) * (1.0 + spread)

    def select_queries(
        self,
        budget: int,
        clusters: Optional[List[Cluster]] = None,
        profiles: Optional[Dict[str, NodeProfile]] = None,
        use_quantilization: bool = True,
        quantile: float = 0.9,
        num_eval_trials: int = 50,
    ) -> List[str]:
        profiles = profiles or self.compute_all_profiles()
        clusters = clusters or self.cluster_nodes(profiles)
        for cluster in clusters:
            cluster.representative_id = self.select_median_rank_representative(cluster, profiles)
            cluster.priority_score = self.compute_cluster_priority(cluster, profiles)
        clusters.sort(key=lambda cluster: cluster.priority_score, reverse=True)
        base_nodes = [
            cluster.representative_id
            for cluster in clusters
            if cluster.representative_id
        ][:budget]
        if len(base_nodes) < budget:
            remaining = (
                node_id for cluster in clusters for node_id in cluster.node_ids
                if node_id not in base_nodes
            )
            base_nodes.extend(list(remaining)[: budget - len(base_nodes)])
        candidate_sets = [base_nodes]
        if use_quantilization and len(clusters) > budget:
            for offset in range(1, min(3, len(clusters) - budget + 1)):
                alternative = [
                    cluster.representative_id
                    for cluster in clusters[offset : offset + budget]
                    if cluster.representative_id
                ]
                if len(alternative) == budget:
                    candidate_sets.append(alternative)
        if not use_quantilization or len(candidate_sets) == 1:
            return candidate_sets[0]
        best_set, best_score = candidate_sets[0], float("inf")
        for query_set in candidate_sets:
            scores = self._evaluate_query_set_quantile(
                query_set, profiles, num_eval_trials
            )
            score = np.percentile(scores, quantile * 100)
            if score < best_score:
                best_set, best_score = query_set, score
        return best_set

    def _evaluate_query_set_quantile(
        self,
        query_nodes: List[str],
        profiles: Dict[str, NodeProfile],
        num_trials: int,
    ) -> List[float]:
        import random

        rng = random.Random(42)
        all_nodes = set(self.graph.thoughts)
        scores: List[float] = []
        for _ in range(num_trials):
            targets: Set[str] = set()
            candidates = all_nodes.copy()
            while not targets and candidates:
                node_id = rng.choice(list(candidates))
                targets.add(node_id)
                profile = profiles.get(node_id)
                if profile:
                    candidates -= profile.pset | profile.rset | {node_id}
            answers = {
                node_id: bool(profiles[node_id].rset & targets)
                for node_id in query_nodes
                if node_id in profiles
            }
            scores.append(
                len(self.update_candidate_set(query_nodes, answers))
            )
        return scores

    def update_candidate_set(
        self,
        query_nodes: List[str],
        answers: Dict[str, bool],
        initial_candidates: Optional[Set[str]] = None,
    ) -> Set[str]:
        candidates = (
            initial_candidates or set(self.graph.thoughts)
        ).copy()
        profiles = self.compute_all_profiles()
        for node_id in query_nodes:
            if node_id not in answers or node_id not in profiles:
                continue
            if answers[node_id]:
                candidates -= profiles[node_id].pset
            else:
                candidates -= profiles[node_id].rset
                candidates.discard(node_id)
        return candidates


__all__ = ["Cluster", "NodeProfile", "QueryOptimizer"]
