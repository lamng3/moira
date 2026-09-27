"""AgentOI: agentic ontology integration with budgeted LLM inference."""

__version__ = "0.1.0"

__all__ = ["OntologySummary", "OntologyWorkspace", "__version__"]


def __getattr__(name: str):
    if name not in {"OntologySummary", "OntologyWorkspace"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from agentoi.workspace import OntologySummary, OntologyWorkspace

    globals()["OntologySummary"] = OntologySummary
    globals()["OntologyWorkspace"] = OntologyWorkspace
    return globals()[name]
