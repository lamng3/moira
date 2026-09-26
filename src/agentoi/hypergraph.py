from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agentoi.application import PipelineConfiguration, run_pipeline
from agentoi.functions.similarity import SimilarityMethod
from agentoi.main import compute_similarity, evaluate_alignment_binary
from agentoi.model_runtime import ModelRuntime


PACKAGE_ROOT = Path(__file__).resolve().parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--terms",
        nargs="*",
        help="List of terms to use with include in user query prompt.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=str(PACKAGE_ROOT / "parser/namespaces/config.json"),
        help="Namespace config",
    )
    parser.add_argument("--version_num", type=str, default="0.1.0")
    parser.add_argument(
        "--ontology",
        nargs="+",
        type=str,
        default=["envo", "sweet"],
        help="Exactly two ontology keys to union (e.g., envo sweet).",
    )
    parser.add_argument("--tool", type=str, default="ontology_term_info")
    parser.add_argument("--top_k", type=int, default=12)
    parser.add_argument("--to_rdf", type=bool, default=True)
    parser.add_argument("--format", type=str, default="ttl")
    parser.add_argument("--load_pickle", action="store_true")
    parser.add_argument("--pickle_filename", type=str, default=None)
    parser.add_argument("--oot_save_local", action="store_true")
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--similarity", action="store_true")
    parser.add_argument("--eval_alignment", action="store_true")
    parser.add_argument("--seed", type=int, default=time.time().__floor__())
    parser.add_argument("--add_noise", action="store_true")
    parser.add_argument("--output", default="results/runs/hypergraph.txt")
    return parser.parse_args(argv)


def _evaluate(
    graph: Any,
    alignment_path: str,
    configuration: PipelineConfiguration,
    runtime: ModelRuntime,
    agent: Any,
) -> dict[str, Any]:
    return evaluate_alignment_binary(
        graph,
        gold_alignment_rdf_path=alignment_path,
        top_k=configuration.top_k,
        target_iri_prefix=None,
        exclusive=True,
        pos_relations={"="},
        neg_relations={"!=", "≠"},
        pos_min_measure=0.5,
        neg_max_measure=0.0,
        verbose=True,
        agent=agent if configuration.refine else None,
        tokenizer=runtime.tokenizer if configuration.refine else None,
    )


def _compute_similarity(
    graph: Any, evaluation: dict[str, Any], alignment_path: str
) -> dict[str, Any]:
    return compute_similarity(
        graph,
        similarity_mode=SimilarityMethod.HYBRID,
        similarity_threshold=None,
        gold_by_source=evaluation.get("gold_by_source"),
        top_ids_by_source=evaluation.get("top_ids_by_source"),
        gold_alignment_rdf_path=alignment_path,
        pos_relations={"="},
        min_measure=None,
    )


def main() -> None:
    load_dotenv()
    args = parse_args()
    configuration = PipelineConfiguration(
        package_root=PACKAGE_ROOT,
        graph_kind="hypergraph",
        ontology_names=tuple(args.ontology),
        config_path=args.config_path,
        version=args.version_num,
        datapaths="data/datapaths.json",
        output=args.output,
        terms=tuple(args.terms or ()),
        tool=args.tool,
        top_k=args.top_k,
        load_pickle=args.load_pickle,
        pickle_filename=args.pickle_filename,
        save_local_traces=args.oot_save_local,
        refine=args.refine,
        similarity=args.similarity,
        evaluate_alignment=args.eval_alignment,
        seed=args.seed,
        noise_percent=50 if args.add_noise else 0,
    )
    run_pipeline(
        configuration,
        evaluate=_evaluate,
        compute_similarity=_compute_similarity,
    )


if __name__ == "__main__":
    main()
