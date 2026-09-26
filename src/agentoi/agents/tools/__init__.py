"""Built-in AgentOI tool providers."""

__all__ = [
    "ontology_info",
    "ontology_term_info",
    "search_knowledge_graph",
    "search_term_context",
]


def __getattr__(name: str):
    if name in {"ontology_info", "ontology_term_info"}:
        from .ontology import ontology_info, ontology_term_info

        return {
            "ontology_info": ontology_info,
            "ontology_term_info": ontology_term_info,
        }[name]
    if name == "search_knowledge_graph":
        from .knowledge_graph import search_knowledge_graph

        return search_knowledge_graph
    if name == "search_term_context":
        from .website import search_term_context

        return search_term_context
    raise AttributeError(name)
