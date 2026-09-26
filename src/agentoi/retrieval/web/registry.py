"""Named web-retrieval plugins."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .base import WebSearchProvider

ProviderFactory = Callable[..., WebSearchProvider]


@dataclass(frozen=True)
class WebSearchPlugin:
    """Developer-facing description of one retrieval backend."""

    name: str
    description: str
    network_access: bool
    environment_variables: tuple[str, ...] = ()


class WebSearchRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, ProviderFactory] = {}
        self._plugins: dict[str, WebSearchPlugin] = {}

    def register(
        self,
        plugin: WebSearchPlugin,
        factory: ProviderFactory,
        *,
        replace: bool = False,
    ) -> None:
        key = self._key(plugin.name)
        if key in self._factories and not replace:
            raise ValueError(f"Web-search provider already registered: {plugin.name!r}")
        self._factories[key] = factory
        self._plugins[key] = plugin

    def create(self, name: str, **options: Any) -> WebSearchProvider:
        key = self._key(name)
        try:
            return self._factories[key](**options)
        except KeyError as error:
            available = ", ".join(self.available())
            raise ValueError(
                f"Unknown web-search provider {name!r}; available: {available}"
            ) from error

    def unregister(self, name: str) -> None:
        key = self._key(name)
        self._factories.pop(key, None)
        self._plugins.pop(key, None)

    def describe(self, name: str) -> WebSearchPlugin:
        try:
            return self._plugins[self._key(name)]
        except KeyError as error:
            raise ValueError(f"Unknown web-search provider {name!r}") from error

    def available(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    @staticmethod
    def _key(name: str) -> str:
        key = name.strip().lower()
        if not key:
            raise ValueError("Provider name cannot be empty")
        return key


registry = WebSearchRegistry()


def _register_builtins() -> None:
    from .google import GoogleCSEProvider
    from .mediawiki import MediaWikiSearchProvider
    from .offline import DatasetSearchProvider, StaticSearchProvider

    registry.register(
        WebSearchPlugin(
            "static",
            "Deterministic counts for tests and offline experiments.",
            False,
        ),
        StaticSearchProvider,
    )
    registry.register(
        WebSearchPlugin(
            "dataset",
            "Deterministic counts generated from a concept graph.",
            False,
        ),
        DatasetSearchProvider,
    )
    registry.register(
        WebSearchPlugin(
            "mediawiki",
            "MediaWiki document counts; no credentials required.",
            True,
        ),
        MediaWikiSearchProvider,
    )
    registry.register(
        WebSearchPlugin(
            "google-cse",
            "Google Programmable Search Engine document counts.",
            True,
            ("GOOGLE_API_KEY", "GOOGLE_CSE_CX"),
        ),
        GoogleCSEProvider,
    )


_register_builtins()


def create_web_search(name: str = "static", **options: Any) -> WebSearchProvider:
    """Create a web-search provider by its visible plugin name."""
    return registry.create(name, **options)


def available_web_searches() -> tuple[str, ...]:
    return registry.available()


def describe_web_search(name: str) -> WebSearchPlugin:
    return registry.describe(name)


def register_web_search(
    plugin: WebSearchPlugin,
    factory: ProviderFactory,
    *,
    replace: bool = False,
) -> None:
    registry.register(plugin, factory, replace=replace)


def unregister_web_search(name: str) -> None:
    registry.unregister(name)
