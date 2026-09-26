"""Shared helpers for relation-reasoning dataset adapters."""

from __future__ import annotations

import re


_REPLACEMENTS = (
    ("->", "_to_"),
    ("<-", "_from_"),
    (">=", "_gte_"),
    ("<=", "_lte_"),
    ("!=", "_ne_"),
    ("=", "_eq_"),
    ("<", "_lt_"),
    (">", "_gt_"),
    ("%", "_pct"),
    ("&", "_and"),
    ("+", "_plus"),
    ("@", "_at"),
    ("/", "_"),
    ("\\", "_"),
    (" ", "_"),
)


def safe_uri_component(value: str) -> str:
    """Convert a dataset label into the package's stable URI component form."""
    result = value
    for old, new in _REPLACEMENTS:
        result = result.replace(old, new)
    result = re.sub(r"""[()[\]{}?# !$,;:'".]""", "", result)
    result = re.sub(r"_+", "_", result).strip("_")
    return result or "unknown_type"
