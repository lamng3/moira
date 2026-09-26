"""Canonical conversation-memory models for Ontology of Thought."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .models import ThoughtType


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: str | datetime | None) -> datetime:
    if isinstance(value, datetime):
        return value
    if not value:
        return _now()
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return _now()


@dataclass
class ThoughtNode:
    id: str = field(default_factory=lambda: str(uuid4()))
    type: ThoughtType = ThoughtType.PROMPT
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=_now)
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    reasoning_depth: int = 0
    confidence: float = 1.0
    tools_used: List[str] = field(default_factory=list)

    def add_child(self, child: "ThoughtNode") -> None:
        if child.id not in self.children_ids:
            self.children_ids.append(child.id)
            child.parent_id = self.id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "type": self.type.value, "content": self.content,
            "metadata": self.metadata, "timestamp": self.timestamp.isoformat(),
            "parent_id": self.parent_id, "children_ids": list(self.children_ids),
            "reasoning_depth": self.reasoning_depth,
            "confidence": self.confidence, "tools_used": list(self.tools_used),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThoughtNode":
        return cls(
            id=data["id"], type=ThoughtType(data["type"]),
            content=data.get("content", ""), metadata=data.get("metadata", {}),
            timestamp=_parse_datetime(data.get("timestamp")),
            parent_id=data.get("parent_id"),
            children_ids=list(data.get("children_ids", [])),
            reasoning_depth=int(data.get("reasoning_depth", 0)),
            confidence=float(data.get("confidence", 1.0)),
            tools_used=list(data.get("tools_used", [])),
        )


@dataclass
class MemoryTrace:
    id: str = field(default_factory=lambda: str(uuid4()))
    nodes: Dict[str, ThoughtNode] = field(default_factory=dict)
    root_node_id: Optional[str] = None
    current_node_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)

    def add_node(self, node: ThoughtNode, parent_id: Optional[str] = None) -> None:
        self.nodes[node.id] = node
        self.updated_at = _now()
        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id].add_child(node)
        if self.root_node_id is None:
            self.root_node_id = node.id
        self.current_node_id = node.id

    def get_node(self, node_id: str) -> Optional[ThoughtNode]:
        return self.nodes.get(node_id)

    def get_children(self, node_id: str) -> List[ThoughtNode]:
        node = self.get_node(node_id)
        return [self.nodes[node_id] for node_id in node.children_ids if node_id in self.nodes] if node else []

    def get_leaves(self) -> List[ThoughtNode]:
        return [node for node in self.nodes.values() if not node.children_ids]

    def get_path_to_root(self, node_id: str) -> List[ThoughtNode]:
        path: List[ThoughtNode] = []
        current: Optional[str] = node_id
        while current:
            node = self.get_node(current)
            if node is None:
                break
            path.append(node)
            current = node.parent_id
        return list(reversed(path))

    def get_conversation_flow(self) -> List[ThoughtNode]:
        return self.get_path_to_root(self.current_node_id) if self.current_node_id else []

    def get_depth(self, node_id: str) -> int:
        return max(0, len(self.get_path_to_root(node_id)) - 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "root_node_id": self.root_node_id,
            "current_node_id": self.current_node_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryTrace":
        trace = cls(
            id=data["id"], root_node_id=data.get("root_node_id"),
            current_node_id=data.get("current_node_id"),
            metadata=data.get("metadata", {}),
            created_at=_parse_datetime(data.get("created_at")),
            updated_at=_parse_datetime(data.get("updated_at")),
        )
        trace.nodes = {
            node.id: node
            for value in (data.get("nodes") or {}).values()
            for node in [ThoughtNode.from_dict(value)]
        }
        return trace


class ConversationTrace:
    def __init__(self, trace_id: Optional[str] = None):
        self.memory_trace = MemoryTrace(id=trace_id or str(uuid4()))
        self.current_reasoning_depth = 0

    def _add(
        self, thought_type: ThoughtType, content: str, *,
        metadata: Optional[Dict[str, Any]] = None, confidence: float = 1.0,
        tools_used: Optional[List[str]] = None,
    ) -> str:
        node = ThoughtNode(
            type=thought_type, content=content, metadata=metadata or {},
            reasoning_depth=self.current_reasoning_depth, confidence=confidence,
            tools_used=tools_used or [],
        )
        self.memory_trace.add_node(node, self.memory_trace.current_node_id)
        return node.id

    def add_prompt(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        return self._add(ThoughtType.PROMPT, content, metadata=metadata)

    def add_response(
        self, content: str, confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        return self._add(
            ThoughtType.RESPONSE, content, metadata=metadata, confidence=confidence
        )

    def add_reasoning(
        self, content: str, depth_increase: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        self.current_reasoning_depth += depth_increase
        return self._add(ThoughtType.REASONING, content, metadata=metadata)

    def add_tool_call(
        self, tool_name: str, args: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        details = {**(metadata or {}), "tool_name": tool_name, "args": args}
        return self._add(
            ThoughtType.TOOL_CALL, f"Tool call: {tool_name}({args})",
            metadata=details, tools_used=[tool_name],
        )

    def add_tool_result(
        self, tool_name: str, result: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        details = {**(metadata or {}), "tool_name": tool_name, "result": result}
        return self._add(
            ThoughtType.TOOL_RESULT, f"Tool result: {tool_name} -> {str(result)[:200]}...",
            metadata=details, tools_used=[tool_name],
        )

    def add_reflection(
        self, content: str, confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        return self._add(
            ThoughtType.REFLECTION, content,
            metadata=metadata, confidence=confidence,
        )

    def step_back_reasoning(self, levels: int = 1) -> None:
        self.current_reasoning_depth = max(0, self.current_reasoning_depth - levels)

    def get_trace(self) -> MemoryTrace:
        return self.memory_trace

    def get_conversation_summary(self) -> str:
        flow = self.memory_trace.get_conversation_flow()
        if not flow:
            return "Empty conversation"
        return "\n".join(
            f"{index}. {'  ' * node.reasoning_depth}[{node.type.value}] {node.content[:100]}..."
            for index, node in enumerate(flow, 1)
        )


__all__ = ["ConversationTrace", "MemoryTrace", "ThoughtNode", "ThoughtType"]
