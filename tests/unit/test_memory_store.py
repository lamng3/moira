import torch

from moira.algorithms.graph import ConceptGraph
from moira.cli import main
from moira.embeddings.encoders.text import TextEmbedding
from moira.memory_store import DEFAULT_TEXT_MODEL, EmbeddingMemory, clear_ontology_memory
from moira.workspace import OntologyWorkspace


ONTOLOGY = """
@prefix ex: <https://example.test/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

ex:ontology a owl:Ontology .
ex:Parent a owl:Class ; rdfs:label "Parent" .
ex:Child a owl:Class ;
    rdfs:subClassOf ex:Parent ;
    rdfs:label "Child" .
"""


def _graph(path):
    workspace = OntologyWorkspace(path, save_memory=False)
    graph = ConceptGraph(nodes=[], edges=[])
    graph.build_from_ontology(workspace.ontology)
    for node in graph.nodes.values():
        node.graph_embedding = torch.ones(4)
        node.text_embedding = torch.ones(4) * 2
        node.embedding = torch.ones(4) * 3
    return graph


def test_saved_embeddings_reload_for_the_same_ontology(tmp_path) -> None:
    ontology = tmp_path / "example.ttl"
    ontology.write_text(ONTOLOGY, encoding="utf-8")
    memory = EmbeddingMemory(ontology, directory=tmp_path / "memory")
    original = _graph(ontology)

    folder = memory.save(original)
    restored = ConceptGraph(nodes=[], edges=[])
    restored.build_from_ontology(OntologyWorkspace(ontology).ontology)

    assert memory.load(restored)
    assert folder.is_dir()
    saved = next(iter(restored.nodes.values()))
    assert torch.equal(saved.embedding, torch.ones(4) * 3)
    assert TextEmbedding.DEFAULT_MODEL_NAME == DEFAULT_TEXT_MODEL


def test_prepare_uses_memory_instead_of_recomputing(tmp_path, monkeypatch) -> None:
    ontology = tmp_path / "example.ttl"
    ontology.write_text(ONTOLOGY, encoding="utf-8")
    memory_dir = tmp_path / "memory"
    EmbeddingMemory(ontology, directory=memory_dir).save(_graph(ontology))

    def fail_recompute(self, alpha=0.5, progress=None):
        raise AssertionError("embeddings were recomputed")

    monkeypatch.setattr(ConceptGraph, "compute_all_embeddings", fail_recompute)
    graph = OntologyWorkspace(
        ontology,
        memory_dir=memory_dir,
        save_memory=False,
    ).prepare()

    assert all(node.embedding is not None for node in graph.nodes.values())


def test_memory_clear_deletes_the_store(tmp_path, capsys) -> None:
    ontology = tmp_path / "example.ttl"
    ontology.write_text(ONTOLOGY, encoding="utf-8")
    memory_dir = tmp_path / "memory"
    EmbeddingMemory(ontology, directory=memory_dir).save(_graph(ontology))

    assert main(["memory", "clear", str(ontology), "--memory-dir", str(memory_dir)]) == 0
    assert clear_ontology_memory(ontology, memory_dir) == []
    assert "Deleted embedding memory" in capsys.readouterr().out
