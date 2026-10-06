from __future__ import annotations

import json
import os
from collections.abc import Callable
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Any, Literal

import torch

from moira.algorithms.graph import ConceptGraph, ConceptHypergraph, Ontology
from moira.algorithms.refinement import OntologyRefiner, RefinementOptions
from moira.model_runtime import (
    ModelRuntime,
    create_application_agent,
    create_model_runtime,
)
from moira.parser import Parser


GraphKind = Literal["graph", "hypergraph"]
EvaluationService = Callable[
    [Any, str, "PipelineConfiguration", ModelRuntime, Any], dict[str, Any]
]
SimilarityService = Callable[
    [Any, dict[str, Any], str], dict[str, Any]
]


@dataclass(frozen=True, slots=True)
class PipelineConfiguration:
    package_root: Path
    graph_kind: GraphKind
    ontology_names: tuple[str, ...]
    config_path: str
    version: str
    datapaths: str
    output: str
    terms: tuple[str, ...]
    tool: str
    top_k: int
    load_pickle: bool
    pickle_filename: str | None
    save_local_traces: bool
    refine: bool
    similarity: bool
    evaluate_alignment: bool
    seed: int
    noise_percent: int
    temperature: float = 0.2
    use_groq: bool = False
    enable_uqlm: bool = False
    uqlm_num_responses: int = 5
    uqlm_confidence_threshold: float = 0.6

    def __post_init__(self) -> None:
        if not 1 <= len(self.ontology_names) <= 2:
            raise ValueError("provide one or two ontology keys")
        if not 0 <= self.noise_percent <= 100:
            raise ValueError("noise_percent must be between 0 and 100")


def human_join(items: tuple[str, ...] | list[str], *, quote: bool = True) -> str:
    values = [str(item).strip() for item in items if str(item).strip()]
    if quote:
        values = [f"'{value}'" for value in values]
    if len(values) < 2:
        return values[0] if values else ""
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"


def build_user_query(tool: str, terms: tuple[str, ...], ontologies: tuple[str, ...]) -> str:
    term_text = human_join(terms)
    ontology_text = human_join(
        [ontology.upper() for ontology in ontologies], quote=False
    )
    return (
        f"Use {tool} to find relations between {term_text} ({ontology_text}), "
        "including part-of and subclass links. Place ONLY and ALL relevant "
        "concepts from concept cluster and relations in the 'term' argument "
        "(as a list). After using the tool, give a summary of the returned terms."
    )


def _load_ontologies(
    configuration: PipelineConfiguration,
) -> tuple[Ontology, dict[str, str] | None]:
    with Path(configuration.datapaths).open(encoding="utf-8") as file:
        catalog: dict[str, dict[str, dict[str, str]]] = json.load(file)

    loaded: list[Ontology] = []
    for name in configuration.ontology_names:
        entry = catalog["rdf"].get(name) or catalog["ttl"].get(name)
        if entry is None:
            raise ValueError(f"Invalid ontology given: {name}")
        loaded.append(
            Parser(
                entry["datapath"],
                config_path=configuration.config_path,
                name=name,
                version=configuration.version,
            ).to_ontology()
        )

    combined_entry: dict[str, str] | None = None
    if len(configuration.ontology_names) == 2:
        combined_name = "-".join(configuration.ontology_names)
        combined_entry = (
            catalog["rdf"].get(combined_name) or catalog["ttl"].get(combined_name)
        )
        if combined_entry:
            loaded.append(
                Parser(
                    combined_entry["datapath"],
                    config_path=configuration.config_path,
                    name=combined_name,
                    version=configuration.version,
                ).to_ontology()
            )
    return Ontology.union_ontologies(loaded), combined_entry


def _prepare_ontology(
    ontology: Ontology, configuration: PipelineConfiguration
) -> Ontology:
    prepared = ontology
    if configuration.refine:
        refiner = OntologyRefiner()
        prepared = refiner.refine(
            ontology, RefinementOptions(), fallback_if_expanded=True
        )
        if any(
            diagnostic.code == "closure_fallback"
            for diagnostic in refiner.last_result.diagnostics
        ):
            print(
                "Number of edges in refined ontology greater than before. "
                "Re-refining with transitive closure disabled..."
            )
        print(f"Number of edges before refinement: {len(ontology.edges)}")
        print(f"Number of edges after refinement: {len(prepared.edges)}")

    if configuration.noise_percent:
        prepared, _ = prepared.apply_bart_noise(
            cfg={
                "p_record": configuration.noise_percent / 100,
                "p_field": 0.8,
                "fields": [
                    "labels",
                    "alt_labels",
                    "related_synonyms",
                    "exact_synonyms",
                ],
            },
            seed=configuration.seed,
        )
    return prepared


def _pickle_filename(configuration: PipelineConfiguration) -> str:
    if configuration.pickle_filename:
        return configuration.pickle_filename
    prefix = (
        "concept_graph"
        if configuration.graph_kind == "graph"
        else "concept_hypergraph"
    )
    return f"{prefix}_{'_'.join(configuration.ontology_names)}.pkl"


def _build_graph(ontology: Ontology, configuration: PipelineConfiguration) -> Any:
    filename = _pickle_filename(configuration)
    if configuration.graph_kind == "graph":
        if configuration.load_pickle:
            return ConceptGraph.load_pickle(filename=filename)
        graph = ConceptGraph(nodes=[], edges=[])
        graph.build_from_ontology(ontology)
    else:
        if configuration.load_pickle:
            return ConceptHypergraph.load_pickle(filename=filename)
        graph = ConceptHypergraph().build_from_ontology(ontology)
    graph.compute_all_embeddings(alpha=0.5)
    graph.to_pickle(filename=filename)
    return graph


def _print_evaluation(result: dict[str, Any]) -> None:
    counts = result["counts"]
    metrics = result["metrics"]
    retrieval = result["positives_only_retrieval"]
    print("=== Binary Alignment Evaluation (with LLM veto for negatives) ===")
    print(
        f"Gold pairs: {counts['|gold|']}  TP={counts['TP']} TN={counts['TN']} "
        f"FP={counts['FP']} FN={counts['FN']}  LLM_checks={counts['llm_checks']}"
    )
    print(
        f"Acc={metrics['accuracy']:.4f}  P={metrics['precision']:.4f} "
        f"R={metrics['recall']:.4f}  Spec={metrics['specificity']:.4f} "
        f"F1={metrics['f1']:.4f}  BalAcc={metrics['balanced_accuracy']:.4f} "
        f"MCC={metrics['mcc']:.4f}"
    )
    print(
        f"[Positives-only] HIT@{retrieval['k']}={retrieval['Hits@k']:.4f}  "
        f"MRR={retrieval['MRR']:.4f}"
    )
    print(f"Embedding avg time: {result['strict_time_avg']:.4f}s")
    print(f"LLM-veto avg time: {result['llm_time_avg']:.4f}s")


def _write_experiment_metrics(
    evaluation: dict[str, Any],
    similarity: dict[str, Any],
) -> None:
    """Export stable scalar results when invoked by the experiment runner."""
    destination = os.environ.get("MOIRA_METRICS_PATH")
    if not destination:
        return
    payload: dict[str, Any] = {}
    if evaluation:
        payload.update(
            {
                "alignment": evaluation.get("metrics", {}),
                "counts": evaluation.get("counts", {}),
                "retrieval": evaluation.get("positives_only_retrieval", {}),
                "timing": {
                    "embedding_average_seconds": evaluation.get(
                        "strict_time_avg", 0.0
                    ),
                    "llm_average_seconds": evaluation.get("llm_time_avg", 0.0),
                },
            }
        )
    if similarity:
        payload["similarity"] = similarity.get(
            "similarity_avgs_over_gold", similarity
        )
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_pipeline(
    configuration: PipelineConfiguration,
    *,
    evaluate: EvaluationService,
    compute_similarity: SimilarityService,
) -> None:
    """Run the shared CLI application pipeline for either graph representation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    output_path = Path(configuration.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output, redirect_stdout(output):
        runtime = create_model_runtime(
            temperature=configuration.temperature,
            use_groq=configuration.use_groq,
            local_do_sample=configuration.graph_kind == "graph",
        )
        agent = create_application_agent(
            runtime,
            package_root=configuration.package_root,
            save_local=configuration.save_local_traces,
        )
        ontology, combined_entry = _load_ontologies(configuration)
        prepared_ontology = _prepare_ontology(ontology, configuration)
        graph = _build_graph(prepared_ontology, configuration)

        evaluation: dict[str, Any] = {}
        similarity: dict[str, Any] = {}
        if configuration.evaluate_alignment and combined_entry:
            alignment_path = (
                combined_entry.get("fullset_datapath")
                or combined_entry["datapath"]
            )
            evaluation = evaluate(
                graph, alignment_path, configuration, runtime, agent
            )
            _print_evaluation(evaluation)

        if configuration.similarity and combined_entry:
            alignment_path = (
                combined_entry.get("fullset_datapath")
                or combined_entry["datapath"]
            )
            similarity = compute_similarity(graph, evaluation, alignment_path)
            print(
                "similarity_avgs_over_gold: "
                f"{similarity['similarity_avgs_over_gold']}"
            )

        if configuration.terms and configuration.tool:
            query = build_user_query(
                configuration.tool,
                configuration.terms,
                configuration.ontology_names,
            )
            guided_prompt = graph.make_prompt_for_query(
                query, list(configuration.terms), k=configuration.top_k, max_edges=60
            )
            print(guided_prompt)
            result = agent.invoke(guided_prompt, return_trace=True)
            pprint(result["trace"])
            pprint(result["result"])
    _write_experiment_metrics(evaluation, similarity)
