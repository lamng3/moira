"""Structural, web, and vector retrieval entry points."""

from agentoi.embeddings.vector_index import (
    available_vector_indexes,
    create_vector_index,
    register_vector_index,
    unregister_vector_index,
)

from .context import (
    ContextCandidate,
    ContextDecision,
    ContextHarness,
    ContextSelection,
    JevContextHarness,
    PassthroughHarness,
    build_context_candidates,
    create_context_harness,
)
from .neighborhood import NeighborhoodSampler, sample_neighborhood
from .web import (
    DatasetSearchProvider,
    GoogleCSEProvider,
    MediaWikiSearchProvider,
    StaticSearchProvider,
    WebSearchPlugin,
    WebSearchProvider,
    available_web_searches,
    create_web_search,
    describe_web_search,
    register_web_search,
    unregister_web_search,
)

__all__ = [
    "ContextCandidate",
    "ContextDecision",
    "ContextHarness",
    "ContextSelection",
    "DatasetSearchProvider",
    "JevContextHarness",
    "PassthroughHarness",
    "GoogleCSEProvider",
    "MediaWikiSearchProvider",
    "NeighborhoodSampler",
    "StaticSearchProvider",
    "WebSearchPlugin",
    "WebSearchProvider",
    "available_vector_indexes",
    "available_web_searches",
    "build_context_candidates",
    "create_context_harness",
    "create_vector_index",
    "create_web_search",
    "describe_web_search",
    "register_vector_index",
    "unregister_vector_index",
    "register_web_search",
    "unregister_web_search",
    "sample_neighborhood",
]
