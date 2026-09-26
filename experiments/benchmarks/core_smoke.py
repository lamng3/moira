"""Deterministic smoke benchmark for core refinement and validation APIs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from agentoi.algorithms.refinement import RelationRefiner, default_rules
from agentoi.validation import evaluate_alignment, validate_matches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("baseline", "refined"), default="refined"
    )
    args = parser.parse_args(argv)
    triples = [
        ("A", "is_a", "B"),
        ("A", "is_a", "B"),
        ("B", "is_a", "C"),
        ("A", "is_a", "C"),
    ]
    if args.mode == "refined":
        output_triples = RelationRefiner().refine(triples).triples
    else:
        output_triples = default_rules().normalize_triples(triples)
    predicted = [(subject, object_) for subject, _, object_ in output_triples]
    gold = [("A", "B"), ("B", "C")]
    alignment = evaluate_alignment(predicted, gold)
    validation = validate_matches([("A", "B"), ("A", "B"), ("B", "C")])

    metrics = {
        "benchmark": {"variant": args.mode},
        "alignment": {
            "f1": alignment.f1_score,
            "false_negatives": alignment.false_negatives,
            "false_positives": alignment.false_positives,
            "precision": alignment.precision,
            "recall": alignment.recall,
            "true_positives": alignment.true_positives,
        },
        "refinement": {
            "input_triples": len(triples),
            "output_triples": len(output_triples),
            "triples": sorted([list(item) for item in output_triples]),
        },
        "validation": {
            "duplicate_count": validation.results[0].duplicate_count,
            "passed": validation.passed,
        },
    }
    destination = os.environ.get("AGENTOI_METRICS_PATH")
    if not destination:
        raise RuntimeError("AGENTOI_METRICS_PATH is required")
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Core smoke benchmark completed ({args.mode}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
