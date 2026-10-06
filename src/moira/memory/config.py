"""Sizes and location for the query cache."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = Path("results/cache")


def _bounded_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    """Caps and eviction for one ontology cache."""

    directory: Path
    hot_entries: int = 32
    short_term_nodes: int = 512
    long_term_entries: int = 1024
    promote_at: int = 3
    eviction: str = "lfu"


def cache_dir(ontology: Path, root: Path | None = None) -> Path:
    """Directory that holds the hot, short-term, and long-term files."""
    digest = hashlib.sha256(ontology.read_bytes()).hexdigest()[:16]
    base = root or Path(os.getenv("MOIRA_CACHE_DIR", str(DEFAULT_ROOT)))
    return base / digest


def config_for(ontology: Path) -> MemoryConfig:
    """Build a config, applying optional environment overrides."""
    override = os.getenv("MOIRA_CACHE_DIR")
    root = Path(override) if override else None
    return MemoryConfig(
        directory=cache_dir(ontology, root),
        hot_entries=_bounded_int("MOIRA_HOT_ENTRIES", 32),
        short_term_nodes=_bounded_int("MOIRA_SHORT_TERM_NODES", 512),
        long_term_entries=_bounded_int("MOIRA_LONG_TERM_ENTRIES", 1024),
        eviction=os.getenv("MOIRA_CACHE_EVICTION", "lfu") or "lfu",
    )
