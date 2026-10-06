"""Compact picture of the query cache for a memory view."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence


def memory_snapshot(
    *,
    hot: Sequence[Mapping[str, object]],
    concepts: Mapping[str, object],
    long_term: Sequence[Mapping[str, object]],
    chat: Mapping[str, object] | None = None,
    names: Callable[[str], str] | None = None,
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
            "label": " / ".join(_named(str(item), names) for item in ids),
            "hits": _hits(item),
        }
        for item in long_term
        if (ids := item.get("concept_ids"))
    ]
    long_rows.sort(key=lambda row: int(row["hits"]), reverse=True)
    return {
        "hot": hot_rows,
        "concepts": _concept_node(concepts, names, root=True),
        "long_term": long_rows,
        "chat": _chat_node(chat or {}),
    }


def _concept_node(
    node: Mapping[str, object],
    names: Callable[[str], str] | None,
    *,
    root: bool = False,
) -> dict[str, object]:
    raw_children = node.get("children") or {}
    children = [
        _concept_node(child, names)
        for child in raw_children.values()
        if isinstance(child, Mapping)
    ]
    children.sort(key=lambda row: int(row["hits"]), reverse=True)
    concept_id = node.get("concept_id")
    return {
        "label": "Concepts" if root or not concept_id else _named(str(concept_id), names),
        "hits": _hits(node),
        "children": children,
    }


def _named(concept_id: str, names: Callable[[str], str] | None) -> str:
    if names is not None:
        label = names(concept_id)
        if label:
            return label
    return short_label(concept_id)


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
