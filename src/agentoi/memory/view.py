"""Compact picture of the query cache for a memory view."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def memory_snapshot(
    *,
    hot: Sequence[Mapping[str, object]],
    concepts: Mapping[str, object],
    long_term: Sequence[Mapping[str, object]],
    chat: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Hot questions, the concept trie, long-term paths, and this chat."""
    hot_rows = [
        {"label": str(item.get("question") or ""), "hits": _hits(item)}
        for item in hot
        if item.get("question")
    ]
    hot_rows.sort(key=lambda row: int(row["hits"]), reverse=True)
    long_rows = [
        {
            "label": " / ".join(short_label(str(item)) for item in ids),
            "hits": _hits(item),
        }
        for item in long_term
        if (ids := item.get("concept_ids"))
    ]
    long_rows.sort(key=lambda row: int(row["hits"]), reverse=True)
    return {
        "hot": hot_rows,
        "concepts": _concept_node(concepts, root=True),
        "long_term": long_rows,
        "chat": _chat_node(chat or {}),
    }


def _concept_node(node: Mapping[str, object], *, root: bool = False) -> dict[str, object]:
    raw_children = node.get("children") or {}
    children = [
        _concept_node(child)
        for child in raw_children.values()
        if isinstance(child, Mapping)
    ]
    children.sort(key=lambda row: int(row["hits"]), reverse=True)
    concept_id = node.get("concept_id")
    return {
        "label": "Concepts" if root or not concept_id else short_label(str(concept_id)),
        "hits": _hits(node),
        "children": children,
    }


def _chat_node(node: Mapping[str, object]) -> dict[str, object]:
    children = [child for child in node.get("children") or [] if isinstance(child, Mapping)]
    if node.get("kind") in {None, "root"}:
        return {
            "label": "This chat",
            "hits": 0,
            "children": [_chat_node(child) for child in children if child.get("kind") == "question"],
        }
    route = next(
        (str(child.get("text") or "") for child in children if child.get("kind") == "route"),
        "",
    )
    label = str(node.get("text") or "")
    if route:
        label = f"{label} · {route}"
    return {
        "label": label,
        "hits": 0,
        "children": [_chat_node(child) for child in children if child.get("kind") == "question"],
    }


def _hits(item: Mapping[str, object]) -> int:
    frequency = item.get("frequency")
    if isinstance(frequency, int) and not isinstance(frequency, bool):
        return frequency
    count = item.get("count")
    if isinstance(count, int) and not isinstance(count, bool):
        return count
    return 0


def short_label(value: str) -> str:
    text = value.rstrip("/").rsplit("/", 1)[-1]
    if "#" in text:
        text = text.rsplit("#", 1)[-1]
    return text or value
