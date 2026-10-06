from __future__ import annotations

import re
from typing import Iterable, Mapping, Optional, Sequence

STOPWORDS = {
    "the", "of", "and", "a", "to", "in", "for", "on", "with", "as", "by",
    "an", "is", "are", "or", "that", "from", "at", "be", "this", "it",
    "its", "into", "within", "which", "such",
}


def first_sentence(text: str, limit: int = 240) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
    return sentence.strip()[:limit]


def top_terms(
    text: str,
    limit: int = 10,
    exclude: Optional[Iterable[str]] = None,
) -> list[str]:
    if not text:
        return []
    excluded = {item.lower() for item in (exclude or [])}
    frequencies: dict[str, int] = {}
    for token in re.findall(r"[a-zA-Z][a-zA-Z\-]{1,}", text):
        token = token.lower()
        if token in STOPWORDS or len(token) < 3 or token in excluded:
            continue
        frequencies[token] = frequencies.get(token, 0) + 1
    return [
        word
        for word, _ in sorted(
            frequencies.items(),
            key=lambda item: (-item[1], item[0]),
        )[:limit]
    ]


def pick_first(
    properties: Mapping[str, object],
    keys: Sequence[str],
) -> Optional[str]:
    for key in keys:
        value = properties.get(key)
        if value is not None and (text := str(value).strip()):
            return text
    return None
