from moira.algorithms.graph import ConceptGraph, Ontology
from moira.validation import OntologyTranslator, ValidationPipeline


TRIPLES = (
    ("Dog", "isA", "Animal"),
    ("Cat", "isA", "Animal"),
    ("Animal", "hasProperty", "Mobility"),
)


def test_translation_uses_moira_graph_models_and_preserves_edges() -> None:
    translator = OntologyTranslator()

    ontology = translator.triples_to_ontology(
        (triple for triple in TRIPLES), name="Animals", version="2.0"
    )
    graph = translator.ontology_to_concept_graph(ontology)

    assert isinstance(ontology, Ontology)
    assert isinstance(graph, ConceptGraph)
    assert ontology.name == "Animals"
    assert ontology.version == "2.0"
    assert len(ontology.nodes) == 4
    assert len(ontology.edges) == 3
    assert len(graph.nodes) == 4
    assert len(graph.edges) == 3


def test_translation_cache_uses_structural_keys() -> None:
    translator = OntologyTranslator()

    first = translator.triples_to_ontology(TRIPLES, name="Animals")
    second = translator.triples_to_ontology(tuple(TRIPLES), name="Animals")

    assert first is second
    assert translator.ontology_to_concept_graph(first) is translator.ontology_to_concept_graph(
        second
    )


def test_offline_validation_pipeline_end_to_end() -> None:
    pipeline = ValidationPipeline()

    result = pipeline.process_ontology(
        (
            ("domestic_dog", "is_a", "animal"),
            ("cat", "type", "animal"),
            ("animal", "has-property", "mobility"),
        ),
        name="NormalizedAnimals",
    )

    assert result.tuples.normalized == (
        ("Domestic Dog", "isA", "Animal"),
        ("Cat", "isA", "Animal"),
        ("Animal", "hasProperty", "Mobility"),
    )
    assert len(result.tuples.evaluations) == 3
    assert result.tuples.statistics.count == 3
    assert result.ontology.name == "NormalizedAnimals"
    assert len(result.concept_graph.nodes) == 4
