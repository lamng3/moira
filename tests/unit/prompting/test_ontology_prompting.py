import pytest

from moira.prompting import (
    OntologyEquivalencePromptBuilder,
    PromptConfig,
    PromptExample,
    PromptResult,
    parse_equivalence_answer,
)


class DescribedClass:
    def __init__(self, description: str) -> None:
        self.description = description

    def describe(self) -> str:
        return self.description


def test_builder_produces_concise_json_prompt_with_examples():
    config = PromptConfig(
        instruction="Compare classes.",
        examples=(
            PromptExample("Cat", "Feline", PromptResult(True, 0.95)),
        ),
    )

    prompt = OntologyEquivalencePromptBuilder(config).build(
        DescribedClass("Lion"), DescribedClass("Big cat")
    )

    assert prompt.startswith("Compare classes.")
    assert prompt.count("Are these ontology classes semantically equivalent?") == 2
    assert '{"answer":"yes","confidence":0.95}' in prompt
    assert "A: Lion\nB: Big cat" in prompt
    assert prompt.endswith(
        'Return only JSON: {"answer":"yes"|"no","confidence":0.0-1.0}'
    )


@pytest.mark.parametrize(
    ("response", "equivalent", "confidence"),
    [
        ('{"answer":"yes","confidence":0.8}', True, 0.8),
        (
            'Result:\n```json\n{"answer": "NO", "confidence": 0}\n```',
            False,
            0.0,
        ),
        (
            'ignored {bad} then {"answer":"yes","confidence":1}',
            True,
            1.0,
        ),
    ],
)
def test_parser_handles_machine_readable_answers(
    response, equivalent, confidence
):
    assert parse_equivalence_answer(response) == PromptResult(
        equivalent, confidence
    )


@pytest.mark.parametrize(
    "response",
    [
        "",
        '{"answer":"maybe","confidence":0.5}',
        '{"answer":"yes"}',
        '{"answer":"yes","confidence":"0.5"}',
        '{"answer":"yes","confidence":1.01}',
        '{"answer":"yes","confidence":-0.01}',
    ],
)
def test_parser_rejects_malformed_or_out_of_range_answers(response):
    with pytest.raises(ValueError):
        parse_equivalence_answer(response)


def test_prompt_result_validates_confidence():
    with pytest.raises(ValueError, match="between 0 and 1"):
        PromptResult(True, float("nan"))
