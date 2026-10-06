"""MOIRA agent APIs."""

__all__ = [
    "Agent",
    "ByteTokenizer",
    "JevModelRouter",
    "ModelRoute",
    "OOTAgentObserver",
    "OOTReconstructor",
    "QueryOptimizer",
    "ThoughtGraph",
    "ToolGateDecision",
    "ToolRiskGate",
    "Trace",
    "TraceStep",
    "create_model_router",
    "create_oot_enhanced_agent",
    "create_tool_gate",
    "ontology_term_info",
    "search_knowledge_graph",
    "search_term_context",
]


def __getattr__(name: str):
    if name == "Agent":
        from .core import Agent

        return Agent
    if name in {"Trace", "TraceStep"}:
        from .runtime.trace import Trace, TraceStep

        return {"Trace": Trace, "TraceStep": TraceStep}[name]
    if name in {"OOTAgentObserver", "create_oot_enhanced_agent"}:
        from .observers import OOTAgentObserver, create_oot_enhanced_agent

        return {
            "OOTAgentObserver": OOTAgentObserver,
            "create_oot_enhanced_agent": create_oot_enhanced_agent,
        }[name]
    if name in {"ByteTokenizer", "ThoughtGraph"}:
        from .ontology_of_thought.graph import ByteTokenizer, ThoughtGraph

        return {"ByteTokenizer": ByteTokenizer, "ThoughtGraph": ThoughtGraph}[name]
    if name == "QueryOptimizer":
        from .ontology_of_thought.optimizer import QueryOptimizer

        return QueryOptimizer
    if name == "OOTReconstructor":
        from .ontology_of_thought.reconstructor import OOTReconstructor

        return OOTReconstructor
    if name in {
        "JevModelRouter",
        "ModelRoute",
        "ToolGateDecision",
        "ToolRiskGate",
        "create_model_router",
        "create_tool_gate",
    }:
        from . import decisions

        return getattr(decisions, name)
    if name in {
        "ontology_term_info",
        "search_knowledge_graph",
        "search_term_context",
    }:
        from . import tools

        return getattr(tools, name)
    raise AttributeError(name)
