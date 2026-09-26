from io import StringIO

import torch

from agentoi.algorithms.graph import ConceptGraph
from agentoi.progress import StderrProgress
from agentoi.workspace import OntologyWorkspace


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


def test_progress_keeps_the_concept_counter_on_one_line() -> None:
    stream = StringIO()
    progress = StderrProgress(stream)

    progress.stage("Building the concept graph from 2 concepts.")
    progress.tick("Embedding concepts: 1/2")
    progress.tick("Embedding concepts: 2/2")
    progress.stage("Embedded 2 concepts.")

    text = stream.getvalue()
    assert "Building the concept graph from 2 concepts.\n" in text
    assert "\rEmbedding concepts: 2/2\n" in text
    assert text.endswith("Embedded 2 concepts.\n")


def test_embedding_reports_each_stage(tmp_path, monkeypatch) -> None:
    ontology = tmp_path / "example.ttl"
    ontology.write_text(ONTOLOGY, encoding="utf-8")
    workspace = OntologyWorkspace(ontology)
    graph = ConceptGraph(nodes=[], edges=[])
    graph.build_from_ontology(workspace.ontology)

    class FakeText:
        embed_dim = 4

        def compute_embedding(self, _node):
            return torch.zeros(4)

    class FakeGraph:
        def __init__(self, source, dimensions, save_path=None, **_kwargs):
            self.embs = {
                node_id: torch.zeros(dimensions) for node_id in source.nodes
            }

    monkeypatch.setattr("agentoi.embeddings.TextEmbedding", FakeText)
    monkeypatch.setattr("agentoi.embeddings.GraphEmbedding", FakeGraph)
    stream = StringIO()

    graph.compute_all_embeddings(progress=StderrProgress(stream))

    text = stream.getvalue()
    assert "Loading the text embedding model" in text
    assert "Learning graph structure embeddings." in text
    assert "Embedding concepts: 1/" in text
    assert "Embedded" in text


def test_ask_reports_retrieval_and_the_model(tmp_path, monkeypatch) -> None:
    ontology = tmp_path / "example.ttl"
    ontology.write_text(ONTOLOGY, encoding="utf-8")

    def skip_embeddings(self, alpha=0.5, progress=None):
        return self

    monkeypatch.setattr(ConceptGraph, "compute_all_embeddings", skip_embeddings)
    monkeypatch.setattr(
        "agentoi.workspace.build_context_candidates",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        ConceptGraph,
        "make_prompt_for_query",
        lambda self, *_args, **_kwargs: "prompt",
    )
    monkeypatch.setattr(
        "agentoi.workspace.create_model_runtime",
        lambda **_kwargs: object(),
    )

    class FakeAgent:
        def invoke(self, _prompt):
            return "heart is part of the cardiovascular system"

    monkeypatch.setattr(
        "agentoi.workspace.create_application_agent",
        lambda *_args, **_kwargs: FakeAgent(),
    )
    stream = StringIO()
    answer = OntologyWorkspace(ontology).ask(
        "What is the heart part of?",
        model="ollama:llama3.1",
        progress=StderrProgress(stream),
    )

    text = stream.getvalue()
    assert answer == "heart is part of the cardiovascular system"
    assert "Building the concept graph from 2 concepts." in text
    assert "Connecting to ollama:llama3.1." in text
    assert "Retrieving concepts related to the question." in text
    assert "Asking ollama:llama3.1. Waiting for the model to answer." in text
    assert text.endswith("Answer ready.\n")
