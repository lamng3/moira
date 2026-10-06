from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from moira.agents.ontology_of_thought.graph import ThoughtGraph
from moira.agents.runtime.duckdb_store import save_run, trace_db_path, trace_store_name
from moira.agents.runtime.persistence import save_run_graph, save_trace_to_file
from moira.agents.runtime.trace import Trace, TraceStep, export_trace
from moira.agents.tools.registry import default_registry_resource

from .events import emit_event
from .tool_executor import (
    FALLBACK_MODULES,
    coerce_tool_call,
    execute_tool_call,
    extract_tool_json,
    load_tools_json,
    preview,
    safe_slug,
)

if TYPE_CHECKING:
    from moira.agents.observers.oot import OOTAgentObserver

logger = logging.getLogger(__name__)


class Agent:
    """Tool-calling agent with runtime traces and Ontology-of-Thought memory."""

    def __init__(
        self,
        llm: Any,
        tools: Optional[List[Any]] = None,
        tools_path: Optional[Union[str, Path]] = None,
        description: str = (
            "You are a helpful assistant who can use the following tools to complete a task."
        ),
        skills: Optional[List[str]] = None,
        observers: Optional[List[Callable[[Dict[str, Any]], None]]] = None,
        verbose: bool = True,
        save_traces: bool = True,
        traces_dir: Optional[Union[str, Path]] = None,
        oot_dir: Optional[Union[str, Path]] = None,
        dynamo_table: Optional[str] = None,
        dynamo_ttl_days: Optional[int] = None,
        save_local: bool = False,
        oot_observer: Optional["OOTAgentObserver"] = None,
        model_router: Any = None,
        tool_gate: Any = None,
    ):
        load_dotenv()
        self.llm = llm
        self.tools = list(tools or [])
        self.description = description
        self.skills = skills or [
            "You can answer the user question with tools",
            "Use ontology_term_info for ontology labels, synonyms, and descriptions",
            "Use search_term_context for general web lookups of a topic",
        ]
        self.verbose = verbose
        self.save_local = save_local
        self.save_traces = save_traces

        if tools_path is None:
            self.tools_path = default_registry_resource()
        else:
            path = Path(tools_path)
            self.tools_path = path if path.is_file() else None
            if self.tools_path is None:
                logger.warning("tools_path is not a file: %s", tools_path)
        self._tool_registry = load_tools_json(self.tools_path)

        self.model_router = model_router
        self.tool_gate = tool_gate
        self.observers = list(observers or [])
        if oot_observer is None:
            from moira.agents.observers.oot import OOTAgentObserver

            oot_observer = OOTAgentObserver(
                save_traces=save_local,
                traces_dir=str(oot_dir) if oot_dir else None,
            )
        self.oot_observer = oot_observer
        if self.oot_observer:
            self.oot_observer.save_traces = save_local
            self.oot_observer._update_traces_dir()

        cache = Path(__file__).resolve().parents[1] / "shared_memory" / ".cache"
        self.traces_dir = Path(traces_dir) if traces_dir else cache
        self.oot_dir = Path(oot_dir) if oot_dir else cache
        if save_local:
            self.traces_dir.mkdir(parents=True, exist_ok=True)
            self.oot_dir.mkdir(parents=True, exist_ok=True)

        self.oot = ThoughtGraph()
        self.dynamo_table = dynamo_table or os.getenv("DYNAMO_TABLE")
        ttl = dynamo_ttl_days if dynamo_ttl_days is not None else os.getenv("DYNAMO_TTL_DAYS")
        self.dynamo_ttl_days = int(ttl) if ttl not in (None, "") else None
        self.dynamo_disabled = os.getenv("MOIRA_DYNAMO_DISABLED", "0") == "1"
        self.last_query_suggestions: Optional[List[Dict[str, Any]]] = None

    def _emit(self, event_type: str, **data: Any) -> None:
        emit_event(
            event_type,
            observers=self.observers,
            oot_observer=self.oot_observer,
            logger=logger,
            **data,
        )

    def prompt_template(self, query: str) -> str:
        return (
            "You are given an ontology developing task and a list of available tools.\n"
            f"- Task: {query}\n"
            f"- Tools list: {json.dumps(self._tool_registry)}\n"
            "------------------------\n"
            "Instructions:\n"
            "- Answer naturally if no tool is required.\n"
            "- For ontology labels, synonyms, descriptions, or aliases, use ontology_term_info.\n"
            "- For a general web lookup, use search_term_context.\n"
            "- When using a tool, return only a JSON object with tool_name, tool_type, "
            "arguments, and module_path.\n"
            "- Do not invent arguments or fields."
        )

    def answer_plain(self, prompt: str) -> str:
        """Answer from retrieved concepts without offering tools."""
        original = self.llm
        try:
            router = self._active_model_router()
            if router is not None:
                fallback = str(getattr(self.llm, "model", "unknown"))
                replacement, _route = router.select(prompt, fallback)
                if replacement is not None:
                    self.llm = replacement
            response = self.llm.invoke([
                SystemMessage(content=(
                    "Answer the QUERY in a few plain sentences. "
                    "Name the concepts that answer it. "
                    "Do not mention headings, numbered evidence, or these instructions. "
                    "Do not call tools, do not return JSON, and do not answer an earlier question."
                )),
                HumanMessage(content=prompt),
            ])
            return getattr(response, "content", str(response)).strip()
        finally:
            self.llm = original

    def _coerce_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        return coerce_tool_call(tool_call, self._tool_registry, FALLBACK_MODULES)

    def _execute_tool(self, tool_call: dict[str, Any]) -> Any:
        tool_name = tool_call["tool_name"]
        module_path = tool_call.get("module_path")
        arguments = tool_call.get("arguments", {}) or {}
        self._emit(
            "tool.exec",
            tool_name=tool_name,
            module_path=module_path,
            args=arguments,
        )
        if self.verbose:
            logger.info(
                "Executing tool %s from %s with args=%s",
                tool_name,
                module_path,
                arguments,
            )
        result = execute_tool_call(
            tool_call,
            self._tool_registry,
            fallbacks=FALLBACK_MODULES,
        )
        self._emit("tool.result", tool_name=tool_name, preview=preview(result, limit=None))
        return result

    def _make_run_dir(self, run_id: str, query_text: str) -> Path:
        run_dir = self.oot_dir / f"{run_id}_{safe_slug(query_text, limit=48)}"
        run_dir.mkdir(parents=True, exist_ok=True)
        index = {
            "run_id": run_id,
            "slug": safe_slug(query_text, limit=48),
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "base_dir": str(run_dir),
        }
        (run_dir / "run_index.json").write_text(
            json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return run_dir

    def _sync_oot_from_observer(self, run_id: str):
        if not self.oot_observer:
            return None
        memory = self.oot_observer.get_latest_memory_trace()
        if memory is not None and memory.id == run_id:
            self.oot = ThoughtGraph.from_memory_trace(memory)
            return memory
        return None

    def _trace_error(
        self, trace: Trace, run_id: str, error: Exception | str, return_trace: bool
    ) -> Any:
        trace.error = str(error)
        self._emit("trace.end", run_id=run_id, error=trace.error)
        self._sync_oot_from_observer(run_id)
        payload = {"error": trace.error}
        if return_trace:
            payload["trace"] = export_trace(trace)
        return payload

    def _summarize_tool_result(
        self, tool_call: dict[str, Any], result: Any, question: str = ""
    ) -> Any:
        """Attach a plain-language answer while retaining the provider's raw result."""
        if not isinstance(result, dict):
            return result
        tool_name = tool_call.get("tool_name")
        if tool_name not in {
            "ontology_term_info", "search_term_context", "search_knowledge_graph"
        }:
            return result
        terms = (tool_call.get("arguments") or {}).get("term", [])
        if isinstance(terms, str):
            try:
                parsed = json.loads(terms)
                terms = parsed if isinstance(parsed, list) else [terms]
            except json.JSONDecodeError:
                terms = [part.strip() for part in terms.replace("|", ",").split(",") if part.strip()]
        context = preview(result, limit=6000)
        response = self.llm.invoke([
            SystemMessage(content=(
                "Answer the user's question in plain sentences for a chat. "
                "Use the tool result as evidence and name the concepts that answer "
                "the question. Do not return JSON, a tool call, or raw records."
            )),
            HumanMessage(content=(
                f"Question: {question}\nOriginal terms: {terms}\nTool result: {context}"
            )),
        ])
        result["per_term_summaries"] = {
            str(term): f"Summary for '{term}' derived from {tool_name} results."
            for term in terms
        }
        result["llm_summary"] = getattr(response, "content", str(response)).strip()
        return result

    def _persist(
        self,
        *,
        run_id: str,
        run_dir: Optional[Path],
        query_text: str,
        messages: Sequence[BaseMessage],
        tool_call: Optional[dict[str, Any]],
        result: Any,
        trace: Trace,
        memory_trace: Any,
    ) -> None:
        if not self.save_traces:
            return
        if trace_store_name() == "duckdb":
            path = trace_db_path()
            save_run(
                path,
                run_id=run_id,
                slug=safe_slug(query_text),
                query=query_text,
                graph=self.oot.to_dict(),
                memory_trace=memory_trace.to_dict() if memory_trace else None,
                start_id=memory_trace.root_node_id if memory_trace else "",
                end_id=memory_trace.current_node_id if memory_trace else "",
            )
            self._emit("duckdb.saved", run_id=run_id, path=str(path))
            return
        if self.save_local and run_dir is not None:
            save_trace_to_file(
                run_dir,
                run_id=run_id,
                model_name=str(getattr(self.llm, "model", "unknown")),
                query_text=query_text,
                messages_preview=preview(messages[-1].content) if messages else None,
                tool_call=tool_call,
                result=result,
                trace_payload=export_trace(trace),
                graph=self.oot.to_dict(),
                memory_trace=memory_trace.to_dict() if memory_trace else None,
                start_id=memory_trace.root_node_id if memory_trace else None,
                end_id=memory_trace.current_node_id if memory_trace else None,
            )
            self._emit("local.saved", run_id=run_id, dir=str(run_dir))
            return
        if not self.dynamo_table or self.dynamo_disabled:
            return
        save_run_graph(
            table_name=self.dynamo_table,
            run_id=run_id,
            slug=safe_slug(query_text),
            graph=self.oot.to_dict(),
            memory_trace=memory_trace.to_dict() if memory_trace else None,
            start_id=memory_trace.root_node_id if memory_trace else "",
            end_id=memory_trace.current_node_id if memory_trace else "",
            ttl_days=self.dynamo_ttl_days,
        )
        self._emit("dynamo.saved", run_id=run_id, table=self.dynamo_table)

    def invoke(
        self,
        query: Union[str, Sequence[BaseMessage]],
        return_trace: bool = False,
        **_: Any,
    ) -> Any:
        run_id = str(uuid4())
        trace = Trace(run_id=run_id)
        build = TraceStep(name="build_prompt")
        run_dir: Optional[Path] = None
        try:
            if isinstance(query, str):
                query_text = query
                messages: Sequence[BaseMessage] = [
                    SystemMessage(
                        content=f"{self.description}\nHere are your skills:\n- "
                        + "\n- ".join(self.skills)
                    ),
                    HumanMessage(content=self.prompt_template(query)),
                ]
                build.data = {"mode": "string_query"}
            else:
                messages = list(query)
                query_text = next(
                    (
                        str(message.content)
                        for message in reversed(messages)
                        if isinstance(message, HumanMessage)
                    ),
                    "user_query",
                )
                build.data = {"mode": "messages"}
            build.status = "ok"
            if self.save_local:
                run_dir = self._make_run_dir(run_id, query_text)
        except Exception as error:
            build.status = "error"
            build.data = {"error": str(error)}
            build.end = time.perf_counter()
            trace.steps.append(build)
            self._emit("trace.start", run_id=run_id, initial_prompt=str(query))
            return self._trace_error(trace, run_id, error, return_trace)
        build.end = time.perf_counter()
        trace.steps.append(build)

        self._emit("trace.start", run_id=run_id, initial_prompt=query_text)
        original_llm = self.llm
        try:
            route = self._apply_model_route(query_text, trace)
            self._emit(
                "llm.request",
                model=str(getattr(self.llm, "model", "unknown")),
                message_preview=messages[-1].content,
            )
            llm_step = TraceStep(name="llm_invoke")
            try:
                response = self.llm.invoke(messages)
                content = getattr(response, "content", str(response))
                llm_step.status = "ok"
                llm_step.data = {
                    "response_preview": preview(content),
                    "response": content,
                }
                if route is not None:
                    llm_step.data["route"] = route.to_dict()
                self._emit("llm.response", preview=content)
            except Exception as error:
                llm_step.status = "error"
                llm_step.data = {"error": str(error)}
                llm_step.end = time.perf_counter()
                trace.steps.append(llm_step)
                return self._trace_error(trace, run_id, error, return_trace)
            llm_step.end = time.perf_counter()
            trace.steps.append(llm_step)

            parse_step = TraceStep(name="parse_tool_call")
            raw_call = extract_tool_json(content)
            tool_call = self._coerce_tool_call(raw_call) if raw_call else None
            parse_step.status = "ok"
            parse_step.data = (
                {"tool_call": tool_call} if tool_call else {"note": "No tool call"}
            )
            parse_step.end = time.perf_counter()
            trace.steps.append(parse_step)
            if tool_call:
                self._emit("tool.parsed", tool_call=tool_call)
            else:
                self._emit("tool.skipped", reason="no_tool_json")

            result: Any = content
            if tool_call:
                tool_step = TraceStep(name="tool_execute")
                try:
                    decision = self._gate_tool_call(tool_call)
                    if decision is not None and decision.blocked:
                        result = decision.refusal()
                        tool_step.status = "blocked"
                        tool_step.data = decision.to_dict()
                        self._emit(
                            "tool.blocked",
                            tool_name=tool_call.get("tool_name"),
                            decision=decision.to_dict(),
                        )
                        self._emit(
                            "tool.result",
                            tool_name=tool_call.get("tool_name"),
                            preview=preview(result, limit=None),
                        )
                    else:
                        result = self._execute_tool(tool_call)
                        result = self._summarize_tool_result(tool_call, result, query_text)
                        tool_step.status = "ok"
                        tool_step.data = {"result_preview": preview(result)}
                except Exception as error:
                    tool_step.status = "error"
                    tool_step.data = {"error": str(error)}
                    trace.error = str(error)
                tool_step.end = time.perf_counter()
                trace.steps.append(tool_step)

            trace.result_preview = preview(result)
            self._emit(
                "trace.end",
                run_id=run_id,
                error=trace.error,
                result_preview=preview(result, limit=None),
            )
            memory_trace = self._sync_oot_from_observer(run_id)
            suggestions = None
            if self.oot_observer:
                try:
                    suggestions = self.oot_observer.suggest_query_nodes()
                    self.last_query_suggestions = suggestions
                except Exception as error:
                    logger.warning(
                        "Query optimization failed for run %s: %s", run_id, error
                    )
            try:
                self._persist(
                    run_id=run_id,
                    run_dir=run_dir,
                    query_text=query_text,
                    messages=messages,
                    tool_call=tool_call,
                    result=result,
                    trace=trace,
                    memory_trace=memory_trace,
                )
            except Exception as error:
                logger.warning("Failed to persist run %s: %s", run_id, error)

            if return_trace:
                return {
                    "result": result,
                    "trace": export_trace(trace),
                    "graph": self.oot.to_dict(),
                    "memory_trace": memory_trace.to_dict() if memory_trace else None,
                    "query_suggestions": suggestions,
                }
            return result
        finally:
            self.llm = original_llm

    def _active_model_router(self) -> Any:
        if self.model_router is not None:
            return self.model_router
        from moira.agents.decisions import create_model_router

        return create_model_router()

    def _active_tool_gate(self) -> Any:
        if self.tool_gate is not None:
            return self.tool_gate
        from moira.agents.decisions import create_tool_gate

        return create_tool_gate()

    def _apply_model_route(self, query_text: str, trace: Trace) -> Any:
        router = self._active_model_router()
        if router is None:
            return None
        fallback = str(getattr(self.llm, "model", "unknown"))
        replacement, route = router.select(query_text, fallback)
        if replacement is not None:
            self.llm = replacement
        step = TraceStep(name="model_route")
        step.status = "ok"
        step.data = route.to_dict()
        step.end = time.perf_counter()
        trace.steps.append(step)
        self._emit("model.routed", **route.to_dict())
        return route

    def _gate_tool_call(self, tool_call: dict[str, Any]) -> Any:
        gate = self._active_tool_gate()
        if gate is None:
            return None
        return gate.assess(tool_call)
