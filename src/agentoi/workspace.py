"""High-level workspace for inspecting and querying one ontology."""

from __future__ import annotations

import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentoi.memory import (
    AgentMemory,
    config_for,
    query_path_from_graph,
)
from agentoi.routing.chat_trie import ChatTrie
from agentoi.algorithms.graph import ConceptGraph, Ontology
from agentoi.algorithms.refinement import OntologyRefiner, RefinementOptions
from agentoi.model_runtime import create_application_agent, create_model_runtime
from agentoi.memory_store import EmbeddingMemory, default_text_model
from agentoi.parser import Parser
from agentoi.progress import Progress, WorkingStatus, computing_status
from agentoi.run_control import QuestionCancelled, RunControl
from agentoi.retrieval import (
    ContextHarness,
    ContextSelection,
    build_context_candidates,
    create_context_harness,
)


@dataclass(frozen=True, slots=True)
class OntologySummary:
    path: Path
    format: str
    name: str
    version: str
    concepts: int
    relations: int


class OntologyWorkspace:
    """Load an ontology once and expose terminal-friendly operations."""

    def __init__(
        self,
        path: str | Path,
        *,
        format: str | None = None,
        refine: bool = False,
        memory_dir: str | Path | None = None,
        save_memory: bool = False,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Ontology not found: {self.path}")
        self.refine = refine
        self.save_memory = save_memory
        self.memory_dir = Path(memory_dir) if memory_dir is not None else None
        self._parser = Parser(self.path, format=format)
        ontology = self._parser.to_ontology()
        self.ontology = self._refine(ontology) if refine else ontology
        self._graph: ConceptGraph | None = None
        self._agent: Any | None = None
        self._model_name: str | None = None
        self._agent_memory: AgentMemory | None = None
        self._router: Any | None = None
        self._router_model_name: str | None = None
        self.last_answer_source: str | None = None
        self.last_route_note: str | None = None
        self.last_route: Any | None = None
        self.last_thought_seconds: float | None = None
        self.last_context_selection: ContextSelection | None = None
        self.chat_trie = ChatTrie()

    @staticmethod
    def _refine(ontology: Ontology) -> Ontology:
        return OntologyRefiner().refine(
            ontology,
            RefinementOptions(),
            fallback_if_expanded=True,
        )

    @property
    def summary(self) -> OntologySummary:
        return OntologySummary(
            path=self.path,
            format=self._parser.format,
            name=self.ontology.name,
            version=self.ontology.version,
            concepts=len(self.ontology.nodes),
            relations=len(self.ontology.edges),
        )

    def prepare(
        self,
        progress: Progress | None = None,
        control: RunControl | None = None,
    ) -> ConceptGraph:
        """Build and embed the concept graph on first use."""
        _check(control)
        if self._graph is None:
            concept_count = len(self.ontology.nodes)
            if progress is not None:
                progress.stage(
                    f"Building the concept graph from {concept_count} concepts."
                )
            graph = ConceptGraph(nodes=[], edges=[])
            graph.build_from_ontology(self.ontology)
            memory = self._embedding_memory()
            if memory.load(graph):
                if progress is not None:
                    progress.stage(computing_status(concept_count, concept_count))
                    progress.stage(
                        f"Loaded {concept_count} concept embeddings from {memory.location()}."
                    )
            else:
                graph.compute_all_embeddings(alpha=0.5, progress=progress, control=control)
                if self.save_memory:
                    folder = memory.save(graph)
                    if progress is not None:
                        progress.stage(
                            f"Saved {concept_count} concept embeddings to {folder}."
                        )
            _check(control)
            self._graph = graph
        return self._graph

    def _replay_routed_question(
        self,
        query: str,
        memory: AgentMemory,
        progress: Progress | None,
        turns: Sequence[tuple[str, str]],
    ) -> str | None:
        """Ask the small router once, then replay when the local policy says so."""
        from agentoi.routing.understand import (
            ActionRoute,
            harness_base_url,
            understand_question,
        )

        cached = [
            str(entry["question"])
            for entry in memory.hot.to_list()
            if isinstance(entry.get("question"), str)
        ]
        if not cached and not turns:
            self.last_route = ActionRoute("retrieve")
            return None
        harness_url = harness_base_url()
        route = understand_question(
            query,
            cached,
            None if harness_url else self._router_llm(),
            turns,
            harness_url=harness_url,
        )
        self.last_route = route
        if route.action != "replay" or not route.question:
            return None
        answer = memory.lookup_question(route.question)
        if not answer:
            return None
        self.last_answer_source = "route"
        self.last_route_note = route.note()
        if progress is not None:
            progress.stage("Using a remembered answer.")
            progress.stage("Answer ready.")
        return answer

    def _router_llm(self) -> Any:
        from agentoi.routing.understand import router_model_name

        name = router_model_name()
        if self._router is None or self._router_model_name != name:
            self._router = create_model_runtime(model_name=name, temperature=0).llm
            self._router_model_name = name
        return self._router

    def memory_view(self) -> dict[str, object]:
        """Hot questions, concept-trie hits, long-term paths, and this chat."""
        from agentoi.memory.view import memory_snapshot

        memory = self._query_memory()
        return memory_snapshot(
            hot=memory.hot.to_list(),
            concepts=memory.short.to_dict(),
            long_term=memory.long.to_list(),
            chat=self.chat_trie.to_dict(),
        )

    def _query_memory(self) -> AgentMemory:
        if self._agent_memory is None:
            self._agent_memory = AgentMemory(config_for(self.path))
        return self._agent_memory

    def _embedding_memory(self) -> EmbeddingMemory:
        kwargs: dict[str, Any] = {
            "ontology": self.path,
            "refine": self.refine,
            "alpha": 0.5,
            "text_model": default_text_model(),
        }
        if self.memory_dir is not None:
            kwargs["directory"] = self.memory_dir
        return EmbeddingMemory(**kwargs)

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        progress: Progress | None = None,
    ) -> list[dict[str, Any]]:
        """Return ontology concepts relevant to natural-language text."""
        graph = self.prepare(progress=progress)
        self._report_query_encoder(graph, progress)
        if progress is not None:
            progress.stage("Retrieving concepts related to the question.")
        matches = graph.nearest_nodes(query, k=top_k)
        return [
            {
                "id": node.id,
                "label": node.members(return_label=True),
                "score": float(score),
            }
            for node, score in matches
        ]

    def ask(
        self,
        query: str,
        *,
        top_k: int = 10,
        model: str | None = None,
        temperature: float = 0.2,
        save_traces: bool = False,
        harness: ContextHarness | None = None,
        web_passages: Sequence[Mapping[str, object]] | None = None,
        include_neighborhood: bool | None = None,
        progress: Progress | None = None,
        show_log: bool = False,
        control: RunControl | None = None,
        turns: Sequence[tuple[str, str] | Mapping[str, str]] | None = None,
    ) -> str:
        """Answer a question using retrieved ontology context and an LLM."""
        self.last_thought_seconds = None
        self.last_answer_source = None
        self.last_route_note = None
        self.last_route = None
        chat_turns = _chat_turns(turns)
        if not chat_turns:
            self.chat_trie = ChatTrie()
        started = time.perf_counter()
        _check(control)
        memory = self._query_memory()
        remembered = memory.consult_question(query)
        hot_answer = _presentable(remembered.answer or "")
        if hot_answer:
            self.last_answer_source = remembered.source
            self.last_thought_seconds = time.perf_counter() - started
            if progress is not None:
                progress.stage("Using a remembered answer.")
                progress.stage("Answer ready.")
            self._record_chat(query, route="hot", concept_ids=(), answer=hot_answer)
            return hot_answer
        reversed_definition = memory.consult_answer(query)
        named = _concept_name(reversed_definition.answer or "", self._graph)
        if (
            named is None
            and (reversed_definition.answer or "").startswith("eq_")
            and self._graph is None
        ):
            named = _concept_name(
                reversed_definition.answer or "",
                self.prepare(progress=progress, control=control),
            )
        if named:
            self.last_answer_source = reversed_definition.source
            self.last_route_note = f"Matched a remembered definition: {named}."
            self.last_thought_seconds = time.perf_counter() - started
            if progress is not None:
                progress.stage("Using a remembered answer.")
                progress.stage("Answer ready.")
            self._record_chat(query, route="reverse", concept_ids=(), answer=named)
            return named
        routed = self._replay_routed_question(query, memory, progress, chat_turns)
        if routed is not None:
            self.last_thought_seconds = time.perf_counter() - started
            self._record_chat(query, route="replay", concept_ids=(), answer=routed)
            return routed
        graph = self.prepare(progress=progress, control=control)
        model_name = self._model_label(model)
        if self._agent is None:
            if progress is not None:
                progress.stage(f"Connecting to {model_name}.")
            runtime = create_model_runtime(
                model_name=model,
                temperature=temperature,
            )
            self._agent = create_application_agent(
                runtime,
                package_root=Path(__file__).resolve().parent,
                save_local=save_traces,
                show_log=show_log,
            )
            self._model_name = model_name
        _attach_model_client(self._agent, control)
        _check(control)
        context_harness = harness or create_context_harness()
        expand_neighborhood = (
            context_harness.expands_context
            if include_neighborhood is None
            else include_neighborhood
        )
        self._report_query_encoder(graph, progress)
        if progress is not None:
            progress.stage("Retrieving concepts related to the question.")
        candidates = build_context_candidates(
            graph,
            query,
            top_k=top_k,
            include_neighborhood=expand_neighborhood,
            web_passages=web_passages,
        )
        selection = context_harness.select(query, candidates)
        self.last_context_selection = selection
        concept_ids = [
            candidate.id
            for candidate in selection.documents
            if candidate.source != "web"
        ]
        path = query_path_from_graph(query, concept_ids, graph)
        remembered = memory.consult_path(path)
        path_answer = _presentable(remembered.answer or "")
        if path_answer:
            self.last_answer_source = remembered.source
            self.last_thought_seconds = time.perf_counter() - started
            if progress is not None:
                progress.stage("Using a remembered answer.")
                progress.stage("Answer ready.")
            self._record_chat(query, route="path", concept_ids=concept_ids, answer=path_answer)
            return path_answer
        _check(control)
        prompt = graph.make_prompt_for_query(
            query,
            [query],
            k=top_k,
            max_edges=60,
            selected_ids=concept_ids,
        )
        prompt = _append_web_context(prompt, selection)
        prompt = _with_chat(prompt, chat_turns)
        if progress is not None:
            progress.stage(
                f"Asking {self._model_name or model_name}. Waiting for the model to answer."
            )
        run_log = getattr(self._agent, "run_log", None)
        if run_log is not None:
            run_log.clear()
        _check(control)
        working = WorkingStatus(enabled=not show_log)
        working.start()
        started = time.perf_counter()
        try:
            try:
                speaker = getattr(self._agent, "answer_plain", None)
                response = speaker(prompt) if callable(speaker) else self._agent.invoke(prompt)
            except QuestionCancelled:
                raise
            except Exception:
                if control is not None and control.cancelled:
                    raise QuestionCancelled() from None
                raise
            if control is not None and control.cancelled:
                raise QuestionCancelled()
        finally:
            working.stop()
            if control is not None and control.cancelled:
                self._agent = None
                self._model_name = None
        self.last_thought_seconds = time.perf_counter() - started
        if progress is not None:
            progress.stage("Answer ready.")
        answer = _usable_prose(_spoken_answer(response))
        self.last_answer_source = "model"
        if answer:
            memory.remember(path, answer)
        else:
            answer = _NO_ANSWER
        self._record_chat(query, route="retrieve", concept_ids=concept_ids, answer=answer)
        return answer

    def _record_chat(
        self,
        question: str,
        *,
        route: str,
        concept_ids: Sequence[str],
        answer: str,
    ) -> None:
        continues = bool(getattr(self.last_route, "continues", False))
        self.chat_trie.record(
            question,
            continues=continues,
            route=route,
            concept_ids=concept_ids,
            answer=answer,
        )


    @staticmethod
    def _report_query_encoder(graph: ConceptGraph, progress: Progress | None) -> None:
        if progress is not None and getattr(graph, "text_encoder", None) is None:
            progress.stage("Loading the text embedding model to encode the question.")

    def _model_label(self, model: str | None) -> str:
        if self._model_name:
            return self._model_name
        return model or os.getenv("LLM_MODEL") or "the configured model"


def _chat_turns(
    turns: Sequence[tuple[str, str] | Mapping[str, str]] | None,
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for item in turns or []:
        if isinstance(item, Mapping):
            question = str(item.get("question") or "").strip()
            answer = str(item.get("answer") or "").strip()
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            question = str(item[0]).strip()
            answer = str(item[1]).strip()
        else:
            continue
        if question and answer:
            rows.append((question, answer))
    return rows[-4:]


def _with_chat(prompt: str, turns: Sequence[tuple[str, str]]) -> str:
    if not turns:
        return prompt
    lines = [prompt, "EARLIER IN THIS CHAT, FOR BACKGROUND ONLY:"]
    for question, answer in turns:
        lines.append(f"Q: {question}")
        lines.append(f"A: {answer}")
    lines.append("Answer the QUERY above. Do not answer an earlier question.")
    return "\n".join(lines)


def _check(control: RunControl | None) -> None:
    if control is not None:
        control.raise_if_cancelled()


def _attach_model_client(agent: Any, control: RunControl | None) -> None:
    """Let cancel close the model connection, including the Ollama HTTP client."""
    if control is None or agent is None:
        return
    client = getattr(getattr(agent, "llm", None), "_client", None)
    close = getattr(client, "close", None)
    if callable(close):
        control.attach_closer(close)


_NO_ANSWER = "The ontology did not yield an answer."


def _concept_name(token: str, graph: Any) -> str | None:
    """The ontology label for a matched concept. Internal equivalence ids stay hidden."""
    nodes = getattr(graph, "nodes", None) if graph is not None else None
    node = nodes.get(token) if isinstance(nodes, dict) else None
    if node is not None:
        for concept in getattr(node, "equiv_concepts", ()):
            ground = getattr(concept, "ground_set", {}) or {}
            found = ground.get("labels") or ground.get("alt_labels") or []
            if found:
                return str(found[0]).strip()
            name = str(getattr(concept, "name", "")).rstrip("/").rsplit("/", 1)[-1]
            if "#" in name:
                name = name.rsplit("#", 1)[-1]
            name = name.replace("_", " ").strip()
            if name and not name.startswith("eq "):
                return name
    if token.startswith("eq_"):
        return None
    return token or None


_ECHO_MARKERS = (
    "relevant concept clusters",
    "key relations",
    "concept clusters",
    "from cluster",
    "(cluster",
)


def _is_tool_monologue(text: str) -> bool:
    """A tool plan is not an ontology answer."""
    folded = text.lower()
    return "tool_name" in folded or "i will use the tools" in folded


def _line_echoes_prompt(line: str) -> bool:
    folded = line.lower()
    if any(marker in folded for marker in _ECHO_MARKERS):
        return True
    return "cluster" in folded and any(character.isdigit() for character in folded)


_SENTENCE = re.compile(r"(?<!\d[.!?])(?<=[.!?])\s+")


def _without_prompt_echo(text: str) -> str:
    """Drop sentences that repeat the evidence headings or cite a numbered cluster."""
    paragraphs: list[str] = []
    for paragraph in text.split("\n\n"):
        lines: list[str] = []
        for line in paragraph.splitlines():
            kept = [
                sentence
                for sentence in _SENTENCE.split(line.strip())
                if sentence and not _line_echoes_prompt(sentence)
            ]
            if kept:
                lines.append(" ".join(kept))
        if lines:
            paragraphs.append("\n".join(lines))
    return "\n\n".join(paragraphs).strip()


def _presentable(text: str) -> str:
    cleaned = _without_prompt_echo(text)
    if not cleaned or _is_tool_monologue(cleaned):
        return ""
    return cleaned


def _usable_prose(text: str) -> str:
    return _presentable(text)


def _spoken_answer(result: Any) -> str:
    """Return chat prose instead of a raw tool payload."""
    if isinstance(result, dict):
        summary = result.get("llm_summary")
        if isinstance(summary, str) and summary.strip():
            return summary.strip()
        reason = result.get("reason")
        if result.get("blocked") and isinstance(reason, str):
            return reason
    text = str(getattr(result, "content", result)).strip()
    if text.startswith("{") and text.endswith("}"):
        return "The lookup returned records, but not a written answer."
    return text


def _append_web_context(prompt: str, selection: ContextSelection) -> str:
    passages = [
        candidate.text
        for candidate in selection.documents
        if candidate.source == "web"
    ]
    if not passages:
        return prompt
    lines = [prompt, "WEB CONTEXT:"]
    lines.extend(f"- {text}" for text in passages)
    return "\n".join(lines)
