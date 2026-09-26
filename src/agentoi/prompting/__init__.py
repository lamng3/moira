"""Public prompting API."""

from agentoi.prompting.ontology import (
    OntologyEquivalencePromptBuilder,
    PromptConfig,
    PromptExample,
    PromptResult,
    PromptTemplate,
    parse_equivalence_answer,
)

__all__ = [
    "OntologyEquivalencePromptBuilder",
    "PromptConfig",
    "PromptExample",
    "PromptResult",
    "PromptTemplate",
    "parse_equivalence_answer",
]
