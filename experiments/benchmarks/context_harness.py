"""Compare passthrough context with a shadow Jev selection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from agentoi.retrieval import ContextCandidate, JevContextHarness, PassthroughHarness
from agentoi.validation import evaluate_alignment

CANDIDATES = (
    ContextCandidate("water", "Fresh water has low dissolved salts.", "concept", 0.91),
    ContextCandidate("liquid", "Water is a liquid.", "neighborhood", 0.4),
    ContextCandidate("geology", "Bedrock is a geological material.", "concept", 0.2),
)
GOLD = {("water", "relevant"), ("liquid", "relevant")}


def _selector(query, candidates, **options):
    del query
    kept = [candidate for candidate in candidates if candidate.id != "geology"]
    shadow = bool(options["shadow"])
    return {
        "status": "shadow" if shadow else "applied",
        "selected_ids": [candidate.id for candidate in kept],
        "documents": list(candidates) if shadow else kept,
        "decisions": [
            {
                "id": candidate.id,
                "probability": candidate.score,
                "selected": candidate.id != "geology",
                "returned": shadow or candidate.id != "geology",
                "reason": "retained" if candidate.id != "geology" else "below_threshold",
            }
            for candidate in candidates
        ],
        "usage": {"total_tokens": 18},
    }


def _quality(predicted_ids: list[str]) -> dict[str, float | int]:
    predicted = {(identifier, "relevant") for identifier in predicted_ids}
    alignment = evaluate_alignment(predicted, GOLD)
    return {
        "f1": alignment.f1_score,
        "false_negatives": alignment.false_negatives,
        "false_positives": alignment.false_positives,
        "precision": alignment.precision,
        "recall": alignment.recall,
        "true_positives": alignment.true_positives,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("passthrough", "shadow"), required=True)
    args = parser.parse_args(argv)
    if args.mode == "passthrough":
        selection = PassthroughHarness().select("freshwater", CANDIDATES)
        predicted = list(selection.proposed_ids)
    else:
        selection = JevContextHarness(selector=_selector, shadow=True).select(
            "freshwater", CANDIDATES
        )
        predicted = list(selection.proposed_ids)

    metrics = {
        "benchmark": {"variant": args.mode},
        "context": selection.metrics(),
        "selection": _quality(predicted),
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
    print(f"Context harness comparison completed ({args.mode}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
