from __future__ import annotations
from typing import Dict, List, Optional, Iterable, Any, Set
from collections import defaultdict, deque
import json, hashlib
from agentoi.agents.ontology_of_thought.models import (
    Edge,
    EdgeType,
    Path,
    Prompt,
    Response,
    Thought,
    ThoughtType,
)
from agentoi.agents.ontology_of_thought.memory import MemoryTrace
import os
import tiktoken

try:
    from transformers import AutoTokenizer as _AutoTokenizer
except ImportError:
    _AutoTokenizer = None  # type: ignore[misc, assignment]


class ByteTokenizer:
    """Fallback when transformers not installed; uses byte length as proxy for token count."""

    def encode(self, text: str):
        return text.encode("utf-8")


def _get_tokenizer():
    llm_model_name = os.getenv("LLM_MODEL") or "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    if llm_model_name.startswith("gpt"):
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return ByteTokenizer()
    if llm_model_name.startswith("gemini"):
        return ByteTokenizer()
    if _AutoTokenizer is not None:
        try:
            return _AutoTokenizer.from_pretrained(llm_model_name)
        except Exception:
            pass
    return ByteTokenizer()


_tokenizer = None

def _byte_len(s: Optional[str]) -> int:
    return len((s or "").encode("utf-8"))

def _token_len(s: Optional[str]) -> int:
    if not s:
        return 0
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = _get_tokenizer()
    return len(_tokenizer.encode(s))

class ThoughtGraph:
    """
    Unified, in-memory thought graph.
    - Primary storage in OOT base types: Thought (nodes) + Edge (directed links).
    - Fast utilities: dedup, tag filtering, BFS path, run-scoped memory trace.
    - Builders: from MemoryTrace, from an agent execution trace.
    - Exporter: prompt/response/edge "view" compatible with PromptNode/ResponseNode layout.
    """

    def __init__(self):
        self.thoughts: Dict[str, Thought] = {}
        self.edges: Dict[str, Edge] = {}
        self.out_adj: Dict[str, List[str]] = defaultdict(list)  # src -> [edge_id]
        self.in_adj: Dict[str, List[str]] = defaultdict(list)   # dst -> [edge_id]
        self._dedup_index: Dict[str, str] = {}  # key -> thought_id

    @staticmethod
    def _dedup_key(question: Optional[str], answer: Optional[str]) -> str:
        h = hashlib.sha256()
        h.update((question or "").encode("utf-8"))
        h.update(b"|")
        h.update((answer or "").encode("utf-8"))
        return h.hexdigest()

    def add_thought(
        self,
        thought: Thought,
        dedup: bool = False,
        merge_tags: bool = True,
        merge_meta: bool = True
    ) -> str:
        if dedup:
            key = self._dedup_key(thought.question, thought.answer)
            if key in self._dedup_index:
                tid = self._dedup_index[key]
                existing = self.thoughts[tid]
                if merge_tags:
                    merged = sorted(set(existing.tags) | set(thought.tags))
                    existing.tags = merged
                if merge_meta:
                    for k, v in (thought.meta or {}).items():
                        existing.meta.setdefault(k, v)
                return tid
            tid = thought.id
            self._dedup_index[key] = tid
            self.thoughts[tid] = thought
            return tid
        # no dedup
        self.thoughts[thought.id] = thought
        return thought.id

    def add_prompt(self, prompt: Prompt, **extra_meta) -> str:
        """add a Prompt as a Thought(PROMPT)."""
        meta = dict(getattr(prompt, "metadata", {}) or {})
        meta.update(extra_meta or {})
        # keep prompt_type in metadata since Thought doesn't have a prompt_type field
        ptype = getattr(prompt, "prompt_type", "user")
        meta.setdefault("prompt_type", ptype)
        meta.setdefault("prompt_bytes", _byte_len(prompt.text))

        t = Thought.prompt(
            question=prompt.text,
            meta=meta,
            reasoning_depth=getattr(prompt, "reasoning_depth", 0),
            confidence=getattr(prompt, "confidence", 1.0),
            tags=list(getattr(prompt, "tags", []) or []),
        )
        return self.add_thought(t)

    def add_response(self, response: Response, **extra_meta) -> str:
        """add a Response as a Thought(RESPONSE/REASONING/TOOL_RESULT)."""
        meta = dict(response.metadata or {})
        meta.update(extra_meta or {})
        rtype = (response.response_type or "assistant").lower()
        tags = list(getattr(response, "tags", []) or [])
        tools = list(getattr(response, "tools_used", []) or [])

        if rtype == "assistant":
            t = Thought.response(
                answer=response.text,
                meta=meta,               
                tools_used=tools,
                reasoning_depth=response.reasoning_depth,
                confidence=response.confidence,
                tags=tags,
            )
        elif rtype == "reasoning":
            t = Thought.reasoning(
                content=response.text,
                depth=0,                       # keep provided depth in reasoning_depth below
                reasoning_depth=response.reasoning_depth,
                meta=meta,                   
                confidence=response.confidence,
                tools_used=tools,
                tags=tags,
            )
        else:  # "tool"
            t = Thought(
                id=response.id,
                question=None,
                answer=None,
                content=response.text,
                type=ThoughtType.TOOL_RESULT,
                tools_used=tools,
                meta=meta,
                reasoning_depth=response.reasoning_depth,
                confidence=response.confidence,
                tags=tags,
            )
        return self.add_thought(t)

    def connect(self, src: str, dst: str, decision: str, **kw) -> str:
        """
        Create a directed edge src -> dst.
        Extra kwargs can include: kind=EdgeType, confidence, note, event, meta, etc.
        """
        e = Edge.new(src, dst, decision, **kw)
        self.edges[e.id] = e
        self.out_adj[src].append(e.id)
        self.in_adj[dst].append(e.id)
        return e.id

    def nodes_by_tag(self, tag: str) -> List[Thought]:
        return [t for t in self.thoughts.values() if tag in (t.tags or [])]

    def nodes_with_tags(self, tags: Iterable[str], mode: str = "any") -> List[Thought]:
        tags = list(tags)
        if not tags:
            return list(self.thoughts.values())
        if mode == "all":
            return [t for t in self.thoughts.values() if all(x in (t.tags or []) for x in tags)]
        return [t for t in self.thoughts.values() if any(x in (t.tags or []) for x in tags)]

    def get_outgoing_edges(self, node_id: str) -> List[Edge]:
        return [self.edges[eid] for eid in self.out_adj.get(node_id, []) if eid in self.edges]

    def get_incoming_edges(self, node_id: str) -> List[Edge]:
        return [self.edges[eid] for eid in self.in_adj.get(node_id, []) if eid in self.edges]

    def find_path(self, start: str, end: str) -> Optional[Path]:
        """BFS shortest path in edge count."""
        if start not in self.thoughts or end not in self.thoughts:
            return None
        q = deque([start])
        prev_edge: Dict[str, Optional[str]] = {start: None}
        prev_node: Dict[str, Optional[str]] = {start: None}
        while q:
            u = q.popleft()
            if u == end:
                break
            for eid in self.out_adj.get(u, []):
                v = self.edges[eid].dst
                if v not in prev_edge:
                    prev_edge[v] = eid
                    prev_node[v] = u
                    q.append(v)
        if end not in prev_edge:
            return None
        # reconstruct
        node_ids, edge_ids = [end], []
        cur = end
        while cur != start:
            eid = prev_edge[cur]
            edge_ids.append(eid)
            cur = prev_node[cur]
            node_ids.append(cur)
        node_ids.reverse()
        edge_ids.reverse()
        return Path(start=start, end=end, node_ids=node_ids, edge_ids=edge_ids)

    def get_run_path(self, run_id: str, prefer_edges: bool = True) -> Optional[Path]:
        """
        Build a run-scoped path (memory trace):
        - Restrict to nodes with meta.run_id == run_id
        - If prefer_edges, also require edges.meta.run_id == run_id
        """
        nodes = [t for t in self.thoughts.values() if (t.meta or {}).get("run_id") == run_id]
        if not nodes:
            return None

        # identify start/end by tags, fallback to time order
        start_nodes = [t for t in nodes if "user_query" in (t.tags or [])]
        end_nodes = [t for t in nodes if "final_result" in (t.tags or [])]

        def _sort_by_time(ts: List[Thought]) -> List[Thought]:
            # created_at is ISO string in our models
            return sorted(ts, key=lambda x: x.created_at)

        start = (start_nodes and _sort_by_time(start_nodes)[0]) or _sort_by_time(nodes)[0]
        end = (end_nodes and _sort_by_time(end_nodes)[-1]) or _sort_by_time(nodes)[-1]

        allowed_nodes = {t.id for t in nodes}
        allowed_edges = {eid for eid, e in self.edges.items() if (e.meta or {}).get("run_id") == run_id}

        q = deque([start.id])
        prev_edge: Dict[str, Optional[str]] = {start.id: None}
        prev_node: Dict[str, Optional[str]] = {start.id: None}
        while q:
            u = q.popleft()
            if u == end.id:
                break
            for eid in self.out_adj.get(u, []):
                if prefer_edges and eid not in allowed_edges:
                    continue
                v = self.edges[eid].dst
                if v not in allowed_nodes:
                    continue
                if v not in prev_edge:
                    prev_edge[v] = eid
                    prev_node[v] = u
                    q.append(v)
        if end.id not in prev_edge:
            return None

        # reconstruct
        node_ids, edge_ids = [end.id], []
        cur = end.id
        while cur != start.id:
            eid = prev_edge[cur]
            edge_ids.append(eid)
            cur = prev_node[cur]
            node_ids.append(cur)
        node_ids.reverse()
        edge_ids.reverse()
        return Path(
            start=start.id,
            end=end.id,
            node_ids=node_ids,
            edge_ids=edge_ids,
            meta={"run_id": run_id},
        )

    def memory_trace(self, start: str, end: str):
        p = self.find_path(start, end)
        return p.to_trace() if p else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thoughts": [t.to_dict() for t in self.thoughts.values()],
            "edges": [e.to_dict() for e in self.edges.values()],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ThoughtGraph":
        g = ThoughtGraph()
        for t in d.get("thoughts", []):
            obj = Thought.from_dict(t) if hasattr(Thought, "from_dict") else Thought(**t)
            g.thoughts[obj.id] = obj
        for e in d.get("edges", []):
            obj = Edge.from_dict(e) if hasattr(Edge, "from_dict") else Edge(**e)
            g.edges[obj.id] = obj
            g.out_adj[obj.src].append(obj.id)
            g.in_adj[obj.dst].append(obj.id)
        return g

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @staticmethod
    def from_json(s: str) -> "ThoughtGraph":
        return ThoughtGraph.from_dict(json.loads(s))

    @classmethod
    def from_memory_trace(cls, mt: MemoryTrace) -> "ThoughtGraph":
        """
        build a ThoughtGraph from a MemoryTrace (ThoughtNode tree).
        """
        g = cls()
        id_map: Dict[str, str] = {}

        # nodes
        for nid, n in mt.nodes.items():
            metadata = dict(n.metadata or {})
            metadata.setdefault("run_id", mt.id)
            tags: List[str] = []
            if n.type == ThoughtType.PROMPT:
                tags.append("user_query")
            if metadata.get("final_result"):
                tags.append("final_result")

            if n.type == ThoughtType.PROMPT:
                m = metadata
                m.setdefault("prompt_bytes", _byte_len(n.content))
                m.setdefault("prompt_tokens", _token_len(n.content))
                t = Thought.prompt(
                    question=n.content,
                    meta=m,
                    reasoning_depth=n.reasoning_depth,
                    confidence=n.confidence,
                )
            elif n.type == ThoughtType.RESPONSE:
                m = metadata
                m.setdefault("response_bytes", _byte_len(n.content))
                t = Thought.response(
                    answer=n.content,
                    meta=m,
                    reasoning_depth=n.reasoning_depth,
                    confidence=n.confidence,
                )
            elif n.type == ThoughtType.REASONING:
                t = Thought.reasoning(
                    content=n.content,
                    depth=0,
                    reasoning_depth=n.reasoning_depth,
                    meta=metadata,
                    confidence=n.confidence,
                )
            elif n.type == ThoughtType.TOOL_CALL:
                t = Thought.tool_call(
                    name=(n.metadata or {}).get("tool_name", "tool"),
                    args=(n.metadata or {}).get("args", {}),
                    meta=metadata,
                )
            elif n.type == ThoughtType.TOOL_RESULT:
                t = Thought.tool_result(
                    name=(n.metadata or {}).get("tool_name", "tool"),
                    result=(n.metadata or {}).get("result"),
                    meta=metadata,
                )
            else:  # REFLECTION or others
                t = Thought.reflection(
                    content=n.content,
                    confidence=n.confidence,
                    meta=metadata,
                )

            t.id = n.id
            t.parent_id = n.parent_id
            t.children_ids = list(n.children_ids)
            t.tags = tags

            # preserve timestamps if present
            if hasattr(n, "timestamp") and n.timestamp:
                try:
                    t.created_at = n.timestamp.isoformat()
                    t.updated_at = t.created_at
                except Exception:
                    pass

            g.add_thought(t)
            id_map[nid] = t.id

        # edges (parent -> child)
        for nid, n in mt.nodes.items():
            if n.parent_id and n.parent_id in id_map and nid in id_map:
                src = id_map[n.parent_id]
                dst = id_map[nid]
                # map MemoryTrace node type to edge kind
                kind = EdgeType.NEXT
                if n.type == ThoughtType.REASONING:
                    kind = EdgeType.JUSTIFIES
                elif n.type in (ThoughtType.TOOL_CALL, ThoughtType.TOOL_RESULT):
                    kind = EdgeType.TOOL_INVOCATION
                elif n.type == ThoughtType.REFLECTION:
                    kind = EdgeType.REFLECTS

                g.connect(
                    src,
                    dst,
                    decision="flow",
                    kind=kind,
                    confidence=getattr(n, "confidence", None),
                    meta={
                        "original_thought_id": nid,
                        "trace_id": mt.id,
                        "run_id": mt.id,
                    },
                    reasoning_delta=max(0, n.reasoning_depth),
                    src_type=str(mt.nodes[n.parent_id].type.value).lower(),
                    dst_type=str(n.type.value).lower(),
                )

        return g

    @classmethod
    def from_agent_trace(cls, agent_trace: Dict[str, Any]) -> "ThoughtGraph":
        """
        Build from a generic agent execution trace:
        expected keys: {"steps": [{"name": str, "data": dict}, ...], "run_id": str?}
        """
        g = cls()
        steps = agent_trace.get("steps", []) or []
        if not steps:
            return g

        prev_id: Optional[str] = None
        run_id = agent_trace.get("run_id")

        for i, step in enumerate(steps):
            name = step.get("name", f"step_{i}")
            data = step.get("data", {}) or {}
            if name == "build_prompt":
                prompt_text = data.get("prompt", "")
                m = {"step": name, **data}
                if run_id:
                    m["run_id"] = run_id
                # NEW: byte length
                m.setdefault("prompt_bytes", _byte_len(prompt_text))
                m.setdefault("prompt_tokens", _token_len(prompt_text))

                t = Thought.prompt(
                    question=data.get("prompt", ""),
                    prompt_type="system",
                    meta=m,
                )
                g.add_thought(t)
                if prev_id:
                    g.connect(prev_id, t.id, decision="flow", kind=EdgeType.NEXT, meta={"run_id": run_id})
                prev_id = t.id

            elif name == "llm_invoke":
                response_text = data.get("prompt", "")
                m = {"step": name, **data}
                if run_id:
                    m["run_id"] = run_id
                # NEW: byte length
                m.setdefault("response_bytes", _byte_len(response_text))
                t = Thought.response(
                    answer=data.get("response", ""),
                    meta=m,
                )
                g.add_thought(t)
                if prev_id:
                    g.connect(prev_id, t.id, decision="flow", kind=EdgeType.NEXT, meta={"run_id": run_id})
                prev_id = t.id

            elif name in ("tool_call", "tool_exec"):
                tool_name = data.get("tool_name", "unknown")
                t = Thought.tool_call(
                    name=tool_name,
                    args=data.get("args", {}),
                    meta={"step": name, **data, **({"run_id": run_id} if run_id else {})},
                )
                g.add_thought(t)
                if prev_id:
                    g.connect(prev_id, t.id, decision="tool", kind=EdgeType.TOOL_INVOCATION, meta={"run_id": run_id})
                prev_id = t.id

            else:
                # fallback as reasoning step
                t = Thought.reasoning(
                    content=str(data)[:200],
                    meta={"step": name, **data, **({"run_id": run_id} if run_id else {})},
                )
                g.add_thought(t)
                if prev_id:
                    g.connect(prev_id, t.id, decision="flow", kind=EdgeType.JUSTIFIES, meta={"run_id": run_id})
                prev_id = t.id

        return g

    def _edge_kind_to_view_type(self, e: Edge) -> str:
        """map Edge.kind -> simple string type used by visual ThoughtGraph."""
        kind = getattr(e, "kind", None)
        if kind == EdgeType.TOOL_INVOCATION or kind == EdgeType.TOOL_RESULT_OF:
            return "tool_call"
        if kind == EdgeType.REFLECTS:
            return "reflection"
        if kind == EdgeType.JUSTIFIES:
            return "reasoning"
        # NEXT/BRANCH/MERGE and unknowns default to 'flow'
        return "flow"

    def to_prompt_response_view(self) -> Dict[str, Any]:
        """
        Export a dict matching the PromptNode/ResponseNode + ThoughtEdge layout:
        {
            "prompt_nodes": {id: {...}}, "response_nodes": {id: {...}}, "edges": {id: {...}}, ...
        }
        """
        prompt_nodes: Dict[str, Dict[str, Any]] = {}
        response_nodes: Dict[str, Dict[str, Any]] = {}
        edges_view: Dict[str, Dict[str, Any]] = {}

        # nodes
        for t in self.thoughts.values():
            ts = t.created_at  # ISO string
            meta = dict(t.meta or {})
            meta.update({"tags": t.tags, "updated_at": t.updated_at})
            if t.type == ThoughtType.PROMPT:
                # use question/content as prompt text
                content = t.question or t.content or ""
                prompt_nodes[t.id] = {
                    "id": t.id,
                    "content": content,
                    "prompt_type": meta.get("prompt_type", meta.get("type", "user")),
                    "metadata": meta,
                    "reasoning_depth": t.reasoning_depth,
                    "confidence": t.confidence,
                    "timestamp": ts,
                }
            else:
                # response / reasoning / tool
                content = t.answer or t.content or ""
                rtype = "assistant"
                if t.type == ThoughtType.REASONING:
                    rtype = "reasoning"
                elif t.type in (ThoughtType.TOOL_CALL, ThoughtType.TOOL_RESULT):
                    rtype = "tool"
                response_nodes[t.id] = {
                    "id": t.id,
                    "content": content,
                    "response_type": rtype,
                    "metadata": meta,
                    "reasoning_depth": t.reasoning_depth,
                    "confidence": t.confidence,
                    "tools_used": list(t.tools_used or []),
                    "timestamp": ts,
                }

        # edges
        for e in self.edges.values():
            edges_view[e.id] = {
                "id": e.id,
                "source_id": e.src,
                "target_id": e.dst,
                "edge_type": self._edge_kind_to_view_type(e),
                "weight": e.confidence if hasattr(e, "confidence") and e.confidence is not None else 1.0,
                "metadata": dict(getattr(e, "meta", {}) or {}),
            }

        return {
            "prompt_nodes": prompt_nodes,
            "response_nodes": response_nodes,
            "edges": edges_view,
        }

    def get_statistics(self) -> Dict[str, Any]:
        total_nodes = len(self.thoughts)
        total_edges = len(self.edges)

        # node type counts
        node_types: Dict[str, int] = defaultdict(int)
        depths: List[int] = []
        for t in self.thoughts.values():
            node_types[str(t.type.value)] += 1
            depths.append(t.reasoning_depth or 0)

        # edge type counts
        edge_types: Dict[str, int] = defaultdict(int)
        for e in self.edges.values():
            edge_types[self._edge_kind_to_view_type(e)] += 1

        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "node_types": dict(node_types),
            "edge_types": dict(edge_types),
            "max_reasoning_depth": max(depths) if depths else 0,
            "avg_reasoning_depth": (sum(depths) / len(depths)) if depths else 0.0,
        }
    
    def select_optimal_queries(
        self,
        budget: int,
        min_cluster_size: int = 2,
        max_clusters: Optional[int] = None,
        use_quantilization: bool = True,
        quantile: float = 0.9,
    ) -> List[str]:
        """
        Select optimal query nodes using HumanGS-style optimization with quantilization.
        
        Args:
            budget: Number of query nodes to select (K)
            min_cluster_size: Minimum nodes per cluster
            max_clusters: Maximum number of clusters (None = auto)
            use_quantilization: Optimize for quantile performance (default: True)
            quantile: Quantile to optimize (0.9 = 90th percentile)
        
        Returns:
            List of node IDs to query, ordered by priority
        """
        from agentoi.agents.ontology_of_thought.optimizer import QueryOptimizer
        optimizer = QueryOptimizer(
            self,
            min_cluster_size=min_cluster_size,
            max_clusters=max_clusters,
        )
        return optimizer.select_queries(
            budget=budget,
            use_quantilization=use_quantilization,
            quantile=quantile,
        )
    
    def update_candidates_from_answers(
        self,
        query_nodes: List[str],
        answers: Dict[str, bool],
        initial_candidates: Optional[Set[str]] = None,
    ) -> Set[str]:
        """
        Update candidate set based on query answers using HumanGS rules.
        
        Args:
            query_nodes: List of node IDs that were queried
            answers: Dict mapping node_id -> True (YES) or False (NO)
            initial_candidates: Starting candidate set (None = all nodes)
        
        Returns:
            Updated candidate set after applying constraints
        """
        from agentoi.agents.ontology_of_thought.optimizer import QueryOptimizer
        optimizer = QueryOptimizer(self)
        return optimizer.update_candidate_set(query_nodes, answers, initial_candidates)
    
    def get_query_nodes_content(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get content and metadata for query nodes.
        
        Args:
            node_ids: List of node IDs to get content for
        
        Returns:
            List of dicts with id, content, type, and metadata for each node
        """
        result = []
        for node_id in node_ids:
            if node_id not in self.thoughts:
                continue
            thought = self.thoughts[node_id]
            content = thought.question or thought.answer or thought.content or ""
            result.append({
                "id": node_id,
                "content": content,
                "type": thought.type.value if hasattr(thought.type, 'value') else str(thought.type),
                "reasoning_depth": thought.reasoning_depth,
                "confidence": thought.confidence,
                "tags": thought.tags or [],
                "metadata": thought.meta or {},
            })
        return result