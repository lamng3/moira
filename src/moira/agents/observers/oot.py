from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

from moira.agents.observers.general import print_observer
from moira.agents.ontology_of_thought.memory import ConversationTrace, MemoryTrace
from moira.agents.ontology_of_thought.graph import ThoughtGraph

logger = logging.getLogger(__name__)

class OOTAgentObserver:
    """
    Observer that listens to Agent events and builds a MemoryTrace in real time.
    At trace end, it also emits a ThoughtGraph (converted from the MemoryTrace) and saves artifacts.
    """

    def __init__(self, save_traces: bool = True, traces_dir: Optional[str] = None):
        self._traces_dir_path = traces_dir
        self.save_traces = save_traces
        self._update_traces_dir()
        self._start_ts: Optional[float] = None
    
    def _update_traces_dir(self):
        """Update traces_dir based on current save_traces setting."""
        self.traces_dir = Path(self._traces_dir_path) if self._traces_dir_path else Path("oot_traces")
        # Only create traces_dir if we might save traces
        if self.save_traces:
            self.traces_dir.mkdir(parents=True, exist_ok=True)

        # current conversation state
        self.current_conversation: Optional[ConversationTrace] = None
        self.current_run_id: Optional[str] = None

        # completed traces
        self.completed_traces: List[ConversationTrace] = []

    def _elapsed(self) -> float:
        return max(0.0, (time.time() - self._start_ts)) if self._start_ts else 0.0

    # Agent._emit can call either `observer(evt)` or `observer.on_event(evt)`
    def __call__(self, evt: Dict[str, Any]) -> None:
        self.on_event(evt)

    def on_event(self, evt: Dict[str, Any]) -> None:
        et = evt.get("event")
        try:
            if et == "trace.start":
                self._handle_trace_start(evt)
            elif et == "model.routed":
                self._handle_model_routed(evt)
            elif et == "llm.request":
                self._handle_llm_request(evt)
            elif et == "llm.response":
                self._handle_llm_response(evt)
            elif et == "tool.parsed":
                self._handle_tool_parsed(evt)
            elif et == "tool.blocked":
                self._handle_tool_blocked(evt)
            elif et == "tool.result":
                self._handle_tool_result(evt)
            elif et == "trace.end":
                self._handle_trace_end(evt)
            # optional: ignore other events silently
        except Exception:
            return

    # ---------------- internal handlers ----------------

    def _handle_trace_start(self, evt: Dict[str, Any]) -> None:
        """Start a new MemoryTrace for this run."""
        self.current_run_id = evt.get("run_id")
        self.current_conversation = ConversationTrace(trace_id=self.current_run_id)
        # prefer emitter timestamp if present; fall back to now
        self._start_ts = float(evt.get("ts")) if evt.get("ts") is not None else time.time()
        # If the emitter ever supplies an initial prompt, record it explicitly.
        initial = evt.get("initial_prompt")
        if initial:
            self.current_conversation.add_prompt(
                initial, metadata={"event": "trace.start", "event_data": evt}
            )

    def _handle_model_routed(self, evt: Dict[str, Any]) -> None:
        """Record the model chosen for this run."""
        if not self.current_conversation:
            return
        name = evt.get("name", "standard")
        model_name = evt.get("model_name", "unknown")
        self.current_conversation.add_reasoning(
            f"Routed to {name} model {model_name}.",
            metadata={
                "event": "model.routed",
                "event_data": evt,
                "elapsed_time": self._elapsed(),
            },
        )

    def _handle_tool_blocked(self, evt: Dict[str, Any]) -> None:
        """Record a tool call that the risk gate refused."""
        if not self.current_conversation:
            return
        decision = evt.get("decision") or {}
        self.current_conversation.add_reflection(
            f"Blocked {evt.get('tool_name', 'unknown')}: {decision.get('reason', '')}",
            confidence=0.0,
            metadata={
                "event": "tool.blocked",
                "event_data": evt,
                "elapsed_time": self._elapsed(),
            },
        )

    def _handle_llm_request(self, evt: Dict[str, Any]) -> None:
        """Record the outbound LLM prompt or intermediate reasoning."""
        if not self.current_conversation:
            return

        message_preview = evt.get("message_preview", "") or ""
        model = evt.get("model", "unknown")

        # If this is the first thing we see, treat as user/system prompt.
        if not self.current_conversation.memory_trace.nodes:
            self.current_conversation.add_prompt(
                message_preview,
                metadata={"model": model, "event": "llm.request", "event_data": evt, "elapsed_time": self._elapsed()},
            )
        else:
            # Otherwise, treat as a reasoning step by the assistant/agent.
            self.current_conversation.add_reasoning(
                f"LLM ({model}) request: {message_preview}",
                metadata={"model": model, "event": "llm.request", "event_data": evt, "elapsed_time": self._elapsed()},
            )

    def _handle_llm_response(self, evt: Dict[str, Any]) -> None:
        """Record the inbound LLM response text."""
        if not self.current_conversation:
            return

        preview = evt.get("preview", "") or ""
        self.current_conversation.add_response(
            preview, metadata={"event": "llm.response", "event_data": evt, "elapsed_time": self._elapsed()}
        )

    def _handle_tool_parsed(self, evt: Dict[str, Any]) -> None:
        """Record a tool call (parsed from the LLM output)."""
        if not self.current_conversation:
            return

        tool_call = evt.get("tool_call", {}) or {}
        tool_name = tool_call.get("tool_name", "unknown")
        args = tool_call.get("arguments", {}) or {}

        self.current_conversation.add_tool_call(
            tool_name, args, metadata={"event": "tool.parsed", "event_data": evt, "elapsed_time": self._elapsed()}
        )

    def _handle_tool_result(self, evt: Dict[str, Any]) -> None:
        """Record a tool execution result (preview)."""
        if not self.current_conversation:
            return

        tool_name = evt.get("tool_name", "unknown")
        preview = evt.get("preview", "") or ""

        self.current_conversation.add_tool_result(
            tool_name, preview, metadata={"event": "tool.result", "event_data": evt, "elapsed_time": self._elapsed()}
        )

    def _handle_trace_end(self, evt: Dict[str, Any]) -> None:
        """Finalize trace: add final result / error (if any), save artifacts, and reset state."""
        if not self.current_conversation:
            return

        # Mark the final result (if provided)
        if evt.get("result_preview"):
            self.current_conversation.add_response(
                f"Final result: {evt['result_preview']}",
                metadata={"final_result": True, "event": "trace.end", "event_data": evt, "elapsed_time": self._elapsed()},
            )

        # Capture errors as reflections
        if evt.get("error"):
            self.current_conversation.add_reflection(
                f"Error: {evt['error']}",
                confidence=0.0,
                metadata={"error": True, "event": "trace.end", "event_data": evt, "elapsed_time": self._elapsed()},
            )

        # Save artifacts
        if self.save_traces:
            self._save_trace(self.current_conversation)

        # Store and reset
        self.completed_traces.append(self.current_conversation)
        self.current_conversation = None
        self.current_run_id = None
        self._start_ts = None

    # ---------------- persistence / export ----------------

    def _save_trace(self, conversation: ConversationTrace) -> None:
        """Persist memory trace + derived graph + summary to disk."""
        memory_trace = conversation.get_trace()

        # Save MemoryTrace JSON
        trace_file = self.traces_dir / f"memory_trace_{memory_trace.id}.json"
        with open(trace_file, "w", encoding="utf-8") as f:
            json.dump(memory_trace.to_dict(), f, ensure_ascii=False, indent=2)

        # Build unified ThoughtGraph from MemoryTrace and save JSON
        thought_graph = ThoughtGraph.from_memory_trace(memory_trace)

        try:
            optimal_ids = thought_graph.select_optimal_queries(budget=2, min_cluster_size=1)
            logger.debug(
                "Optimal query node ids for trace %s: %s",
                memory_trace.id,
                optimal_ids,
            )
        except Exception as e:
            logger.debug("Query optimizer failed for trace %s: %s", memory_trace.id, e)

        graph_file = self.traces_dir / f"thought_graph_{memory_trace.id}.json"
        with open(graph_file, "w", encoding="utf-8") as f:
            json.dump(thought_graph.to_dict(), f, ensure_ascii=False, indent=2)

        # Human-readable summary
        summary_file = self.traces_dir / f"summary_{memory_trace.id}.txt"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(f"Conversation Summary for {memory_trace.id}\n")
            f.write("=" * 60 + "\n\n")
            f.write(conversation.get_conversation_summary())
            f.write("\n\nGraph Statistics:\n")
            stats = thought_graph.get_statistics()
            for k, v in stats.items():
                f.write(f"- {k}: {v}\n")

            # Query optimizer suggestions
            try:
                q_ids = thought_graph.select_optimal_queries(budget=5, min_cluster_size=2)
                q_nodes = thought_graph.get_query_nodes_content(q_ids)
                f.write("\nSuggested query nodes (via query optimizer):\n")
                for node in q_nodes:
                    preview = (node.get("content") or "")[:120].replace("\n", " ")
                    f.write(f"  - {node['id']}: {preview}\n")
            except Exception as e:
                f.write(f"\n[Query optimizer error: {e}]\n")

    # ---------------- convenience APIs ----------------

    def get_latest_trace(self) -> Optional[ConversationTrace]:
        """Return the most recent completed ConversationTrace (if any)."""
        return self.completed_traces[-1] if self.completed_traces else None

    def get_latest_memory_trace(self) -> Optional[MemoryTrace]:
        """Return the canonical memory for the most recently completed run."""
        latest = self.get_latest_trace()
        return latest.get_trace() if latest else None

    def get_latest_thought_graph(self) -> Optional[ThoughtGraph]:
        """Derive a graph view from the canonical conversation memory."""
        memory = self.get_latest_memory_trace()
        return ThoughtGraph.from_memory_trace(memory) if memory else None
    
    def suggest_query_nodes(self, budget: int = 5, min_cluster_size: int = 2) -> Optional[List[Dict[str, Any]]]:
        """
        Suggest optimal query nodes for debugging/analysis using query optimization.
        
        Args:
            budget: Number of query nodes to suggest (K)
            min_cluster_size: Minimum nodes per cluster
        
        Returns:
            List of query node info dicts, or None if no graph available
        """
        graph = self.get_latest_thought_graph()
        if not graph or len(graph.thoughts) == 0:
            return None
        query_node_ids = graph.select_optimal_queries(budget=budget, min_cluster_size=min_cluster_size)
        return graph.get_query_nodes_content(query_node_ids)

    def analyze_conversation_patterns(self) -> Dict[str, Any]:
        """Aggregate simple stats across captured conversations."""
        if not self.completed_traces:
            return {"message": "No traces to analyze"}

        total_conversations = len(self.completed_traces)
        total_nodes = 0
        tools_used: set[str] = set()
        reasoning_depths: List[int] = []
        convo_lengths: List[int] = []

        for conv in self.completed_traces:
            mt = conv.get_trace()
            n_nodes = len(mt.nodes)
            total_nodes += n_nodes
            convo_lengths.append(n_nodes)
            for node in mt.nodes.values():
                for t in (node.tools_used or []):
                    tools_used.add(t)
                reasoning_depths.append(node.reasoning_depth or 0)

        return {
            "total_conversations": total_conversations,
            "total_nodes": total_nodes,
            "avg_conversation_length": total_nodes / total_conversations if total_conversations else 0.0,
            "max_conversation_length": max(convo_lengths) if convo_lengths else 0,
            "min_conversation_length": min(convo_lengths) if convo_lengths else 0,
            "unique_tools_used": len(tools_used),
            "tools_used": sorted(tools_used),
            "max_reasoning_depth": max(reasoning_depths) if reasoning_depths else 0,
            "avg_reasoning_depth": (sum(reasoning_depths) / len(reasoning_depths)) if reasoning_depths else 0.0,
        }

def create_oot_enhanced_agent(
    llm: Any,
    tools_path: Optional[str | Path] = None,
    traces_dir: str = "oot_traces",
    **kwargs: Any,
) -> Any:
    """
    Factory: attach OOTAgentObserver to a standard Agent.
    - Keeps any existing observers (defaulting to `print_observer`)
    - Also sets `agent.oot_observer` so Agent can call `on_event`.
    """
    # Create OOT observer with save_traces=True initially
    # The Agent constructor will override this with save_local value
    oot_observer = OOTAgentObserver(save_traces=True, traces_dir=traces_dir)
    if kwargs.get("observers") is None:
        kwargs["observers"] = [print_observer]
    kwargs["oot_observer"] = oot_observer
    
    # Pass traces_dir as oot_dir to Agent constructor so it uses the same directory
    kwargs["oot_dir"] = traces_dir

    # Local import keeps the observer independent of agent orchestration.
    from moira.agents import Agent

    agent = Agent(llm=llm, tools_path=tools_path, **kwargs)
    return agent