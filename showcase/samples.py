"""Export the playground's sample ontologies to showcase/samples.json, through Aegis's own OWL loader.

Each sample is {name, about, labels: {concept: label}, edges: [[child, parent], ...]}, subClassOf only.
Run from the moira repository root with aegis (and rdflib) available:

    uv run python showcase/samples.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1]
sys.path.insert(0, str(SOURCE / "aegis" / "src"))
sys.path.insert(0, str(HERE.parent / "examples"))

from aegis.owl import load_labels, load_owl  # noqa: E402

FILES = {
    "EBSD": ("OntoCheck/SupplementaryMaterials/Ontologies/EBSD.ttl", "Electron backscatter diffraction (materials science)"),
    "MatProc": ("OntoCheck/SupplementaryMaterials/Ontologies/MatProc.ttl", "Materials processing"),
    "XRD": ("OntoCheck/SupplementaryMaterials/Ontologies/XRD.ttl", "X-ray diffraction"),
    "Mouse anatomy": ("moira/data/MouseHuman/mouse.owl", "Adult mouse anatomy, 2,744 classes"),
}


def export(path: Path) -> dict:
    loaded = load_owl(path)
    labels = load_labels(path, loaded)
    edges = sorted([s, o] for s, p, o in loaded.ontology.relations if p == "subClassOf")
    concepts = sorted(loaded.ontology.concepts)
    return {"labels": {c: labels.get(c) or c for c in concepts}, "edges": edges}


def main() -> None:
    import shared_memory_workflow as demo

    out = [{
        "name": "Materials",
        "about": "The small materials ontology from the recorded run",
        "labels": {c: c for c in sorted(demo.MATERIALS.concepts)},
        "edges": sorted([s, o] for s, p, o in demo.MATERIALS.relations if p == "subClassOf"),
    }]
    for name, (rel, about) in FILES.items():
        path = SOURCE / rel
        if path.is_file():
            out.append({"name": name, "about": about, **export(path)})
            print(f"{name}: {len(out[-1]['labels'])} concepts, {len(out[-1]['edges'])} subClassOf edges")
        else:
            print(f"{name}: {path} not found, skipped")
    (HERE / "samples.json").write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
