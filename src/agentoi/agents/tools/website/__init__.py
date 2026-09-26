"""Website lookup provider."""

__all__ = ["search_term_context"]


def __getattr__(name: str):
    if name == "search_term_context":
        from .website_lookup import search_term_context

        return search_term_context
    raise AttributeError(name)
