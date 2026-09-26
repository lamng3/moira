"""Web-search retrieval plugins."""

from .base import WebSearchProvider
from .google import GoogleCSEProvider
from .mediawiki import MediaWikiSearchProvider
from .offline import DatasetSearchProvider, StaticSearchProvider, VocabularyStats
from .registry import (
    WebSearchPlugin,
    available_web_searches,
    create_web_search,
    describe_web_search,
    register_web_search,
    unregister_web_search,
)

__all__ = [
    "DatasetSearchProvider",
    "GoogleCSEProvider",
    "MediaWikiSearchProvider",
    "StaticSearchProvider",
    "VocabularyStats",
    "WebSearchPlugin",
    "WebSearchProvider",
    "available_web_searches",
    "create_web_search",
    "describe_web_search",
    "register_web_search",
    "unregister_web_search",
]
