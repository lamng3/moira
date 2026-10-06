"""Saved concept-graph embeddings for later ontology sessions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from moira.algorithms.graph import ConceptGraph

MEMORY_VERSION = 1
DEFAULT_MEMORY_DIR = Path("results/memory")
def default_text_model() -> str:
    from moira.embeddings.encoders.text import TextEmbedding

    return TextEmbedding.DEFAULT_MODEL_NAME


DEFAULT_TEXT_MODEL = "Qwen/Qwen3-Embedding-0.6B"


@dataclass(frozen=True, slots=True)
class EmbeddingMemory:
    """One ontology's fused, graph, and text embeddings on disk."""

    ontology: Path
    directory: Path = DEFAULT_MEMORY_DIR
    refine: bool = False
    alpha: float = 0.5
    text_model: str = DEFAULT_TEXT_MODEL

    def location(self) -> Path:
        digest = _file_sha256(self.ontology)[:16]
        stem = "".join(
            character if character.isalnum() or character in "-_" else "-"
            for character in self.ontology.stem
        ).strip("-") or "ontology"
        suffix = "refined" if self.refine else "plain"
        return self.directory / f"{stem}-{suffix}-{digest}"

    def exists(self) -> bool:
        folder = self.location()
        return (folder / "manifest.json").is_file() and (folder / "embeddings.pt").is_file()

    def save(self, graph: ConceptGraph) -> Path:
        nodes = list(graph.nodes.values())
        if not nodes or any(node.embedding is None for node in nodes):
            raise ValueError("Concept graph has no embeddings to save")
        folder = self.location()
        folder.mkdir(parents=True, exist_ok=True)
        ids = [node.id for node in nodes]
        (folder / "manifest.json").write_text(
            json.dumps(
                {
                    "version": MEMORY_VERSION,
                    "ontology": str(self.ontology.resolve()),
                    "sha256": _file_sha256(self.ontology),
                    "refine": self.refine,
                    "alpha": self.alpha,
                    "text_model": self.text_model,
                    "concepts": len(ids),
                    "ids": ids,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        torch.save(
            {
                "graph": torch.stack([node.graph_embedding for node in nodes]),
                "text": torch.stack([node.text_embedding for node in nodes]),
                "fused": torch.stack([node.embedding for node in nodes]),
            },
            folder / "embeddings.pt",
        )
        return folder

    def load(self, graph: ConceptGraph) -> bool:
        """Restore embeddings when the saved ontology and node ids still match."""
        folder = self.location()
        manifest_path = folder / "manifest.json"
        weights_path = folder / "embeddings.pt"
        if not manifest_path.is_file() or not weights_path.is_file():
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not _manifest_matches(manifest, self, graph):
            return False
        weights = torch.load(weights_path, map_location="cpu", weights_only=True)
        ids = list(manifest["ids"])
        for index, node_id in enumerate(ids):
            node = graph.nodes.get(node_id)
            if node is None:
                return False
            node.graph_embedding = weights["graph"][index]
            node.text_embedding = weights["text"][index]
            node.embedding = weights["fused"][index]
        return all(node.embedding is not None for node in graph.nodes.values())


def clear_ontology_memory(
    ontology: Path,
    directory: Path = DEFAULT_MEMORY_DIR,
) -> list[Path]:
    """Delete every saved embedding store for this ontology path."""
    if not directory.is_dir():
        return []
    resolved = str(ontology.expanduser().resolve())
    removed: list[Path] = []
    for manifest_path in directory.glob("*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if manifest.get("ontology") != resolved:
            continue
        folder = manifest_path.parent
        for child in folder.iterdir():
            child.unlink()
        folder.rmdir()
        removed.append(folder)
    return removed


def _manifest_matches(
    manifest: dict[str, object],
    memory: EmbeddingMemory,
    graph: ConceptGraph,
) -> bool:
    if manifest.get("version") != MEMORY_VERSION:
        return False
    if manifest.get("sha256") != _file_sha256(memory.ontology):
        return False
    if manifest.get("refine") is not memory.refine:
        return False
    if manifest.get("alpha") != memory.alpha:
        return False
    if manifest.get("text_model") != memory.text_model:
        return False
    ids = manifest.get("ids")
    if not isinstance(ids, list) or set(ids) != set(graph.nodes):
        return False
    return True


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
