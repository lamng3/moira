"""High-level workspace for inspecting and querying one ontology."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentoi.memory import (
    AgentMemory,
    cached_prefix_note,
    config_for,
    query_path_from_graph,
)
from agentoi.algorithms.graph import ConceptGraph, Ontology
from agentoi.algorithms.refinement import OntologyRefiner, RefinementOptions
from agentoi.model_runtime import create_application_agent, create_model_runtime
from agentoi.memory_store import EmbeddingMemory, default_text_model
from agentoi.parser import Parser
from agentoi.progress import Progress, WorkingStatus
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
        self.last_answer_source: str | None = None
        self.last_context_selection: ContextSelection | None = None

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

    def prepare(self, progress: Progress | None = None) -> ConceptGraph:
        """Build and embed the concept graph on first use."""
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
                    progress.stage(
                        f"Loaded {concept_count} concept embeddings from {memory.location()}."
                    )
            else:
                graph.compute_all_embeddings(alpha=0.5, progress=progress)
                if self.save_memory:
                    folder = memory.save(graph)
                    if progress is not None:
                        progress.stage(
                            f"Saved {concept_count} concept embeddings to {folder}."
                        )
            self._graph = graph
        return self._graph

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
    ) -> str:
        """Answer a question using retrieved ontology context and an LLM."""
        memory = self._query_memory()
        remembered = memory.consult_question(query)
        if remembered.answer:
            self.last_answer_source = remembered.source
            if progress is not None:
                progress.stage("Using a remembered answer.")
                progress.stage("Answer ready.")
            return remembered.answer
        graph = self.prepare(progress=progress)
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
        if remembered.answer:
            self.last_answer_source = remembered.source
            if progress is not None:
                progress.stage("Using a remembered answer.")
                progress.stage("Answer ready.")
            return remembered.answer
        prompt = graph.make_prompt_for_query(
            query,
            [query],
            k=top_k,
            max_edges=60,
            selected_ids=concept_ids,
        )
        prompt = _append_web_context(prompt, selection)
        if remembered.prefix:
            labels = _concept_labels(graph, remembered.prefix)
            note = cached_prefix_note(labels)
            if note:
                prompt = f"{prompt}\n{note}"
            if progress is not None:
                progress.stage(
                    f"Using {len(remembered.prefix)} concepts remembered from earlier questions."
                )
        if progress is not None:
            progress.stage(
                f"Asking {self._model_name or model_name}. Waiting for the model to answer."
            )
        run_log = getattr(self._agent, "run_log", None)
        if run_log is not None:
            run_log.clear()
        working = WorkingStatus(enabled=not show_log)
        working.start()
        try:
            response = self._agent.invoke(prompt)
        finally:
            working.stop()
        if progress is not None:
            progress.stage("Answer ready.")
        answer = _spoken_answer(response)
        self.last_answer_source = "model"
        memory.remember(path, answer)
        return answer


    @staticmethod
    def _report_query_encoder(graph: ConceptGraph, progress: Progress | None) -> None:
        if progress is not None and getattr(graph, "text_encoder", None) is None:
            progress.stage("Loading the text embedding model to encode the question.")

    def _model_label(self, model: str | None) -> str:
        if self._model_name:
            return self._model_name
        return model or os.getenv("LLM_MODEL") or "the configured model"


def _concept_labels(graph: ConceptGraph, concept_ids: tuple[str, ...]) -> list[str]:
    labels: list[str] = []
    for concept_id in concept_ids:
        node = graph.nodes.get(concept_id)
        if node is None:
            labels.append(concept_id)
        else:
            labels.append(node.members(return_label=True))
    return labels


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
