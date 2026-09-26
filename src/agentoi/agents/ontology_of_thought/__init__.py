"""Ontology-of-Thought models, memory, and graph views."""

from .memory import ConversationTrace, MemoryTrace, ThoughtNode
from .models import Edge, EdgeType, Path, Prompt, Response, Thought, ThoughtType

__all__ = [
    "ConversationTrace",
    "Edge",
    "EdgeType",
    "MemoryTrace",
    "Path",
    "Prompt",
    "QueryOptimizer",
    "Response",
    "Thought",
    "ThoughtGraph",
    "ThoughtNode",
    "ThoughtType",
    "Cluster",
    "NodeProfile",
]


def __getattr__(name: str):
    if name == "ThoughtGraph":
        from .graph import ThoughtGraph

        return ThoughtGraph
    if name in {"Cluster", "NodeProfile", "QueryOptimizer"}:
        from .optimizer import Cluster, NodeProfile, QueryOptimizer

        return {
            "Cluster": Cluster,
            "NodeProfile": NodeProfile,
            "QueryOptimizer": QueryOptimizer,
        }[name]
    raise AttributeError(name)
