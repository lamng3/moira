import pytest

from moira.agents import Agent
from moira.algorithms.graph import Ontology
from moira.preprocess.prepare_configuration import (
    OntologyMatchingConfiguration,
    build_configuration_w,
)


def similarity(left: object, right: object) -> float:
    return 1.0 if left == right else 0.0


def make_agent() -> Agent:
    return object.__new__(Agent)


def test_build_configuration_w_materializes_validated_components() -> None:
    source = Ontology(name="source")
    target = Ontology(name="target")
    model = make_agent()

    configuration = build_configuration_w(
        source,
        target,
        (function for function in [similarity]),
        (agent for agent in [model]),
        0.75,
    )

    assert configuration == OntologyMatchingConfiguration(
        source_ontology=source,
        target_ontology=target,
        similarity_functions=(similarity,),
        models=(model,),
        threshold=0.75,
    )


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan")])
def test_build_configuration_w_rejects_out_of_range_threshold(
    threshold: float,
) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        build_configuration_w(
            Ontology(),
            Ontology(),
            [similarity],
            [make_agent()],
            threshold,
        )


def test_build_configuration_w_requires_distinct_ontologies() -> None:
    ontology = Ontology()
    with pytest.raises(ValueError, match="distinct"):
        build_configuration_w(
            ontology,
            ontology,
            [similarity],
            [make_agent()],
            0.5,
        )


@pytest.mark.parametrize(
    ("similarities", "models", "message"),
    [
        ([], [make_agent()], "similarity function"),
        ([similarity], [], "model"),
        ([object()], [make_agent()], "callables"),
        ([similarity], [object()], "Agent"),
    ],
)
def test_build_configuration_w_validates_components(
    similarities: list[object],
    models: list[object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        build_configuration_w(  # type: ignore[arg-type]
            Ontology(),
            Ontology(),
            similarities,
            models,
            0.5,
        )
