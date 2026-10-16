"""MOIRA's answer quality and cost per model, on mouse anatomy hierarchy questions. See README.md."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import rdflib  # noqa: E402

from moira.costs import METER  # noqa: E402
from moira.workspace import OntologyWorkspace  # noqa: E402

ONTOLOGY = ROOT / "data" / "MouseHuman" / "mouse.owl"


def questions(path: Path, n: int) -> list[dict]:
    """Concepts with 3 to 8 direct subclasses, in label order, with their subclasses' labels as gold."""
    graph = rdflib.Graph()
    graph.parse(str(path))
    label = {s: str(o) for s, o in graph.subject_objects(rdflib.RDFS.label)}
    children: dict = {}
    for child, parent in graph.subject_objects(rdflib.RDFS.subClassOf):
        if isinstance(parent, rdflib.URIRef) and child in label and parent in label:
            children.setdefault(parent, []).append(label[child])
    picked = sorted((label[p], sorted(c)) for p, c in children.items() if 3 <= len(c) <= 8)
    step = max(1, len(picked) // n)
    return [{"concept": name, "gold": gold} for name, gold in picked[::step][:n]]


def recall(answer: str, gold: list[str]) -> float:
    text = re.sub(r"\s+", " ", answer.lower())
    return sum(g.lower() in text for g in gold) / len(gold)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=["ollama:llama3.1"])
    parser.add_argument("--questions", type=int, default=8)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    workspace = OntologyWorkspace(ONTOLOGY, save_memory=False)
    items = questions(ONTOLOGY, args.questions)
    rows = []
    for model in args.models:
        METER.reset()
        started = time.time()
        answers = []
        for item in items:
            question = f"What kinds of {item['concept']} are there?"
            answer = workspace.ask(question, model=model, top_k=10, temperature=0.0)
            answers.append({"question": question, "gold": item["gold"], "answer": answer,
                            "recall": round(recall(answer, item["gold"]), 4)})
        cost = METER.summary()
        row = {
            "model": model,
            "questions": len(items),
            "recall": round(sum(a["recall"] for a in answers) / len(answers), 4),
            "wall_seconds": round(time.time() - started, 1),
            "cost": cost,
            "per_question": {
                "calls": round(cost["calls"] / len(items), 2),
                "input_tokens": round(cost["input_tokens"] / len(items)),
                "output_tokens": round(cost["output_tokens"] / len(items)),
                "model_seconds": round(cost["model_seconds"] / len(items), 2),
            },
            "answers": answers,
        }
        rows.append(row)
        print(f"{model}: recall {row['recall']:.3f}; per question {row['per_question']}; "
              f"cost {cost['cost_usd'] if cost['cost_usd'] is not None else 'local'}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
