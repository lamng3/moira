from dataclasses import FrozenInstanceError

import pytest

from agentoi.validation import (
    NormalizationRule,
    TupleEvaluator,
    TupleNormalizer,
)


def test_default_normalization() -> None:
    normalizer = TupleNormalizer()

    assert normalizer.normalize(("domestic_dog", "is_a", "animal")) == (
        "Domestic Dog",
        "isA",
        "Animal",
    )
    assert normalizer.normalize(("car", "has-property", "mobility")) == (
        "Car",
        "hasProperty",
        "Mobility",
    )


def test_custom_rule_order_and_cache_invalidation() -> None:
    normalizer = TupleNormalizer(rules=())
    value = ("A", "type", "B")
    assert normalizer.normalize(value) == value

    normalizer.add_rule(
        NormalizationRule(
            name="type_to_isa",
            condition=lambda item: item[1] == "type",
            transform=lambda item: (item[0], "isA", item[2]),
            priority=100,
        )
    )

    assert normalizer.normalize(value) == ("A", "isA", "B")


def test_rule_errors_are_not_swallowed() -> None:
    def fail(_: tuple[str, str, str]) -> bool:
        raise RuntimeError("invalid rule")

    normalizer = TupleNormalizer(
        [NormalizationRule("failing", fail, lambda item: item)]
    )

    with pytest.raises(RuntimeError, match="invalid rule"):
        normalizer.normalize(("A", "isA", "B"))


@pytest.mark.parametrize(
    ("value", "exception"),
    [
        (("A", "B"), ValueError),
        (("A", "isA", 3), TypeError),
    ],
)
def test_invalid_tuple_is_rejected(value: tuple[object, ...], exception: type[Exception]) -> None:
    with pytest.raises(exception):
        TupleEvaluator().evaluate(value)


def test_evaluation_is_cached_with_tuple_key_and_immutable() -> None:
    evaluator = TupleEvaluator()
    value = ("Animal", "sameAs", "Animal")

    first = evaluator.evaluate(value)
    second = evaluator.evaluate(tuple(value))

    assert first is second
    assert first.overall_score > evaluator.evaluate(
        ("Animal", "differentFrom", "Animal")
    ).overall_score
    with pytest.raises(FrozenInstanceError):
        first.overall_score = 0.0


def test_batch_statistics_include_empty_input() -> None:
    evaluator = TupleEvaluator()

    assert evaluator.statistics([]).count == 0
    results = evaluator.evaluate_all(
        [("Dog", "isA", "Animal"), ("Animal", "sameAs", "Animal")]
    )
    statistics = evaluator.statistics(results)

    assert statistics.count == 2
    assert 0.0 <= statistics.mean_score <= 1.0
    assert 0.0 <= statistics.mean_confidence <= 1.0
