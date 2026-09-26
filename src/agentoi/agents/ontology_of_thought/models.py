from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import re
from typing import Any, Dict, List, Optional
from uuid import uuid4


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


class ThoughtType(Enum):
    """Canonical type classification shared by every OOT representation."""

    PROMPT = "prompt"
    RESPONSE = "response"
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    REFLECTION = "reflection"


@dataclass
class Thought:
    """A graph-level thought derived from a conversation-memory node."""

    id: str
    question: Optional[str] = None
    answer: Optional[str] = None
    content: Optional[str] = None
    type: ThoughtType = ThoughtType.RESPONSE
    tags: List[str] = field(default_factory=list)
    refs: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    reasoning_depth: int = 0
    tools_used: List[str] = field(default_factory=list)
    event: Optional[Dict[str, Any]] = None
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=_iso_now)
    updated_at: str = field(default_factory=_iso_now)

    @staticmethod
    def new(question: str, answer: str, **kwargs: Any) -> "Thought":
        thought_type = kwargs.pop(
            "type",
            ThoughtType.RESPONSE if answer is not None else ThoughtType.PROMPT,
        )
        return Thought(
            id=str(uuid4()),
            question=question,
            answer=answer,
            type=thought_type,
            **kwargs,
        )

    @classmethod
    def prompt(cls, question: str, **kwargs: Any) -> "Thought":
        return cls(
            id=str(uuid4()),
            type=ThoughtType.PROMPT,
            question=question,
            **kwargs,
        )

    @classmethod
    def response(cls, answer: str, **kwargs: Any) -> "Thought":
        return cls(
            id=str(uuid4()),
            type=ThoughtType.RESPONSE,
            answer=answer,
            **kwargs,
        )

    @classmethod
    def reasoning(
        cls, content: str, depth: int = 1, **kwargs: Any
    ) -> "Thought":
        return cls(
            id=str(uuid4()),
            type=ThoughtType.REASONING,
            content=content,
            reasoning_depth=kwargs.pop("reasoning_depth", 0) + depth,
            **kwargs,
        )

    @classmethod
    def tool_call(
        cls, name: str, args: Dict[str, Any], **kwargs: Any
    ) -> "Thought":
        tools_used = list(kwargs.pop("tools_used", []))
        if name not in tools_used:
            tools_used.append(name)
        return cls(
            id=str(uuid4()),
            type=ThoughtType.TOOL_CALL,
            content=f"Tool call: {name}({args})",
            tools_used=tools_used,
            event={"name": name, "args": args},
            **kwargs,
        )

    @classmethod
    def tool_result(cls, name: str, result: Any, **kwargs: Any) -> "Thought":
        tools_used = list(kwargs.pop("tools_used", []))
        if name not in tools_used:
            tools_used.append(name)
        preview = str(result)
        if len(preview) > 200:
            preview = preview[:200] + "..."
        return cls(
            id=str(uuid4()),
            type=ThoughtType.TOOL_RESULT,
            content=f"Tool result: {name} -> {preview}",
            tools_used=tools_used,
            event={"name": name, "result": result},
            **kwargs,
        )

    @classmethod
    def reflection(
        cls, content: str, confidence: float = 1.0, **kwargs: Any
    ) -> "Thought":
        return cls(
            id=str(uuid4()),
            type=ThoughtType.REFLECTION,
            content=content,
            confidence=confidence,
            **kwargs,
        )

    def add_child_id(self, child_id: str) -> None:
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)
            self.touch()

    def link_child(self, child: "Thought") -> None:
        self.add_child_id(child.id)
        child.parent_id = self.id
        child.touch()

    def touch(self) -> None:
        self.updated_at = _iso_now()

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["type"] = self.type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Thought":
        values = dict(data)
        if isinstance(values.get("type"), str):
            values["type"] = ThoughtType(values["type"])
        return cls(**values)

class EdgeType(Enum):
    NEXT = "next"
    JUSTIFIES = "justifies"
    CONTRADICTS = "contradicts"
    REFINES = "refines"
    BRANCH = "branch"
    MERGE = "merge"
    TOOL_INVOCATION = "tool_invocation"
    TOOL_RESULT_OF = "tool_result_of"
    REFLECTS = "reflects"


@dataclass
class Edge:
    id: str
    src: str
    dst: str
    decision: str
    kind: EdgeType = EdgeType.NEXT
    src_type: Optional[str] = None
    dst_type: Optional[str] = None
    order_index: Optional[int] = None
    reasoning_delta: Optional[int] = None
    note: str = ""
    confidence: Optional[float] = None
    event: Optional[Dict[str, Any]] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_iso_now)
    updated_at: str = field(default_factory=_iso_now)

    @staticmethod
    def new(src: str, dst: str, decision: str, **kwargs: Any) -> "Edge":
        return Edge(id=str(uuid4()), src=src, dst=dst, decision=decision, **kwargs)

    def touch(self) -> None:
        self.updated_at = _iso_now()

    def set_event(self, **event: Any) -> None:
        self.event = (self.event or {}) | event
        self.touch()

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Edge":
        values = dict(data)
        if isinstance(values.get("kind"), str):
            values["kind"] = EdgeType(values["kind"])
        return cls(**values)


class PathType(Enum):
    MAIN = "main"
    BRANCH = "branch"
    TOOL_CHAIN = "tool_chain"
    HYPOTHESIS = "hypothesis"
    RETROSPECT = "retrospect"


class PathStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"


@dataclass
class Path:
    id: str = field(default_factory=lambda: str(uuid4()))
    start: str = ""
    end: str = ""
    node_ids: List[str] = field(default_factory=list)
    edge_ids: List[str] = field(default_factory=list)
    kind: PathType = PathType.MAIN
    status: PathStatus = PathStatus.OPEN
    score: Optional[float] = None
    confidence: Optional[float] = None
    tools: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_iso_now)
    updated_at: str = field(default_factory=_iso_now)

    def to_trace(self) -> Dict[str, Any]:
        return {
            "id": self.id, "start": self.start, "end": self.end,
            "sequence": self.node_ids, "edges": self.edge_ids,
            "kind": self.kind.value, "status": self.status.value, "meta": self.meta,
        }

    def append(self, node_id: str, edge_id: Optional[str] = None) -> None:
        if not self.node_ids:
            self.start = node_id
        self.node_ids.append(node_id)
        if edge_id is not None:
            self.edge_ids.append(edge_id)
        self.end = node_id
        self.touch()

    def extend_tools(self, tool_name: str) -> None:
        if tool_name and tool_name not in self.tools:
            self.tools.append(tool_name)
            self.touch()

    def touch(self) -> None:
        self.updated_at = _iso_now()

    def length(self) -> int:
        return len(self.node_ids)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Path":
        values = dict(data)
        if isinstance(values.get("kind"), str):
            values["kind"] = PathType(values["kind"])
        if isinstance(values.get("status"), str):
            values["status"] = PathStatus(values["status"])
        return cls(**values)


_VARIABLE_PATTERNS = (
    re.compile(r"\{\{\s*([a-zA-Z_]\w*)\s*\}\}"),
    re.compile(r"\{\s*([a-zA-Z_]\w*)\s*\}"),
)


def _extract_variables(text: str) -> List[str]:
    return sorted({
        match.group(1)
        for pattern in _VARIABLE_PATTERNS
        for match in pattern.finditer(text or "")
    })


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


@dataclass
class Prompt:
    id: str
    name: str
    text: str
    variables: List[str] = field(default_factory=list)
    description: str = ""
    tags: List[str] = field(default_factory=list)
    prompt_type: str = "user"
    metadata: Dict[str, Any] = field(default_factory=dict)
    reasoning_depth: int = 0
    confidence: float = 1.0
    created_at: str = field(default_factory=_iso_now)
    updated_at: str = field(default_factory=_iso_now)

    @staticmethod
    def new(name: str, text: str, **kwargs: Any) -> "Prompt":
        variables = kwargs.pop("variables", None)
        return Prompt(
            id=str(uuid4()), name=name, text=text,
            variables=_extract_variables(text) if variables is None else variables,
            **kwargs,
        )

    def touch(self) -> None:
        self.updated_at = _iso_now()

    def infer_variables(self) -> List[str]:
        self.variables = _extract_variables(self.text)
        self.touch()
        return self.variables

    def render(self, **kwargs: Any) -> str:
        return (self.text or "").format_map(_SafeDict(**kwargs))

    def to_graph_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "content": self.text, "prompt_type": self.prompt_type,
            "metadata": self.metadata | {
                "name": self.name, "description": self.description,
                "tags": self.tags, "variables": self.variables,
                "created_at": self.created_at, "updated_at": self.updated_at,
                "prompt_bytes": len(self.text.encode("utf-8")),
            },
            "reasoning_depth": self.reasoning_depth,
            "confidence": self.confidence, "timestamp": self.created_at,
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Prompt":
        return cls(**data)


@dataclass
class Response:
    id: str
    text: str
    response_type: str = "assistant"
    tools_used: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    reasoning_depth: int = 0
    confidence: float = 1.0
    created_at: str = field(default_factory=_iso_now)
    updated_at: str = field(default_factory=_iso_now)

    @staticmethod
    def new(text: str, **kwargs: Any) -> "Response":
        return Response(id=str(uuid4()), text=text, **kwargs)

    @classmethod
    def assistant(cls, text: str, **kwargs: Any) -> "Response":
        return cls(id=str(uuid4()), text=text, response_type="assistant", **kwargs)

    @classmethod
    def reasoning(cls, text: str, depth: int = 1, **kwargs: Any) -> "Response":
        reasoning_depth = kwargs.pop("reasoning_depth", 0) + depth
        return cls(
            id=str(uuid4()), text=text, response_type="reasoning",
            reasoning_depth=reasoning_depth, **kwargs,
        )

    @classmethod
    def tool(
        cls, text: str, tool_name: Optional[str] = None, **kwargs: Any
    ) -> "Response":
        tools = list(kwargs.pop("tools_used", []))
        if tool_name and tool_name not in tools:
            tools.append(tool_name)
        return cls(
            id=str(uuid4()), text=text, response_type="tool",
            tools_used=tools, **kwargs,
        )

    def touch(self) -> None:
        self.updated_at = _iso_now()

    def add_tool(self, tool_name: str) -> None:
        if tool_name and tool_name not in self.tools_used:
            self.tools_used.append(tool_name)
            self.touch()

    def to_graph_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "content": self.text,
            "response_type": self.response_type,
            "metadata": self.metadata | {
                "tags": self.tags, "created_at": self.created_at,
                "updated_at": self.updated_at,
            },
            "reasoning_depth": self.reasoning_depth,
            "confidence": self.confidence, "tools_used": list(self.tools_used),
            "timestamp": self.created_at,
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Response":
        return cls(**data)

__all__ = [
    "Edge",
    "EdgeType",
    "Path",
    "PathStatus",
    "PathType",
    "Prompt",
    "Response",
    "Thought",
    "ThoughtType",
]
