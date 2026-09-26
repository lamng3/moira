"""Structural neighborhood retrieval."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from agentoi.algorithms.graph import ConceptGraph, EquivalentClass


@dataclass(frozen=True)
class NeighborhoodSampler:
    """Collect parents, children, and siblings within a hop limit."""

    max_hops: int = 2
    include_center: bool = False

    def sample(
        self,
        graph: ConceptGraph,
        center: EquivalentClass,
    ) -> list[EquivalentClass]:
        if center.id not in graph.nodes:
            raise KeyError(f"Unknown center node: {center.id}")
        if self.max_hops < 0:
            raise ValueError("max_hops cannot be negative")

        visited = {center.id}
        selected = {center} if self.include_center else set()
        queue = deque([(center, 0)])

        while queue:
            node, depth = queue.popleft()
            if depth >= self.max_hops:
                continue
            parents = graph.parents_of(node)
            neighbors = parents + graph.children_of(node)
            for parent in parents:
                neighbors.extend(graph.children_of(parent))

            for neighbor in neighbors:
                if neighbor.id in visited:
                    continue
                visited.add(neighbor.id)
                selected.add(neighbor)
                queue.append((neighbor, depth + 1))

        return sorted(selected, key=lambda node: node.id)


def sample_neighborhood(
    graph: ConceptGraph,
    center: EquivalentClass,
    max_hops: int = 2,
) -> list[EquivalentClass]:
    """Return a deterministic structural neighborhood."""
    return NeighborhoodSampler(max_hops=max_hops).sample(graph, center)
