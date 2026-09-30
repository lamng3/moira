"""Prefix trie of one chat. The root is a virtual node, not a question."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass
class ChatNode:
    kind: str
    text: str = ""
    children: list[ChatNode] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "text": self.text,
            "children": [child.to_dict() for child in self.children],
        }


class ChatTrie:
    """Questions hang on the root, or under the previous question when they continue it."""

    def __init__(self) -> None:
        self.root = ChatNode("root")
        self._latest: ChatNode | None = None

    def record(
        self,
        question: str,
        *,
        continues: bool,
        route: str,
        concept_ids: Sequence[str],
        answer: str,
    ) -> ChatNode:
        parent = self._latest if continues and self._latest is not None else self.root
        node = ChatNode("question", question)
        node.children.append(ChatNode("route", route))
        if concept_ids:
            node.children.append(ChatNode("concepts", ", ".join(concept_ids)))
        node.children.append(ChatNode("answer", answer))
        parent.children.append(node)
        self._latest = node
        return node

    def to_dict(self) -> dict[str, object]:
        return self.root.to_dict()
