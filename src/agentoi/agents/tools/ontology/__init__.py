"""Ontology lookup provider."""

__all__ = ["ontology_term_info"]


def __getattr__(name: str):
    if name == "ontology_term_info":
        from .ontology_lookup import ontology_term_info

        return ontology_term_info
    raise AttributeError(name)
