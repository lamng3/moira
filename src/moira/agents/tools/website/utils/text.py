from __future__ import annotations
import re
from typing import List, Optional

from moira.agents.tools.common.text import (
    first_sentence as _first_sentence,
    top_terms,
)


def first_sentence(text: str, limit: int = 400) -> str:
    return _first_sentence(text, limit)


def dedupe(seq: List[str]) -> List[str]:
    """stable de-duplication, preserving order."""
    seen, out = set(), []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def mine_synonyms(text: str, term: str) -> List[str]:
    """heuristic synonym/alias extraction from prose around `term`."""
    if not text:
        return []
    found: List[str] = []
    patterns = [
        r"also known as ([^.;\n]+)",
        r"\baka\b ([^.;\n]+)",
        r"other names? include ([^.;\n]+)",
        r"alternatively called ([^.;\n]+)",
        r"sometimes called ([^.;\n]+)",
    ]
    for p in patterns:
        for m in re.finditer(p, text, flags=re.I):
            chunk = m.group(1)
            for c in re.split(r",|;|\bor\b|/|\\", chunk):
                s = c.strip(" ()[]{}\"'").lower()
                if s and s != term.lower():
                    found.append(s)

    # parenthetical aliases: "Term (Alias, Abbrev)"
    for m in re.finditer(rf"\b{re.escape(term)}\b\s*\(([^)]+)\)", text, flags=re.I):
        for c in re.split(r",|;|\bor\b|/|\\", m.group(1)):
            s = c.strip(" ()[]{}\"'").lower()
            if s and s != term.lower():
                found.append(s)

    return dedupe(found)

def top_phrases(text: str, limit: int = 12, exclude: Optional[List[str]] = None) -> List[str]:
    """light unigram keyword extractor that keeps hyphens."""
    return top_terms(text, limit=limit, exclude=exclude)
