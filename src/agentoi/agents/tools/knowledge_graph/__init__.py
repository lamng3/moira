"""Knowledge-graph lookup provider."""

__all__ = ["search_knowledge_graph"]


def __getattr__(name: str):
    if name == "search_knowledge_graph":
        from .knowledge_graph_lookup import search_knowledge_graph

        return search_knowledge_graph
    raise AttributeError(name)
