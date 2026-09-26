"""Public validation API for ontology alignments and tuples."""

from .metrics import AlignmentMetrics, evaluate_alignment
from .pipeline import (
    OntologyPipelineResult,
    OntologyTranslator,
    PipelineResult,
    ValidationPipeline,
)
from .rules import (
    DuplicateMatchRule,
    RuleResult,
    ValidationReport,
    ValidationSeverity,
    validate_matches,
)
from .tuples import (
    EvaluationStatistics,
    NormalizationRule,
    TupleEvaluation,
    TupleEvaluator,
    TupleNormalizer,
)

__all__ = [
    "AlignmentMetrics",
    "DuplicateMatchRule",
    "EvaluationStatistics",
    "NormalizationRule",
    "OntologyPipelineResult",
    "OntologyTranslator",
    "PipelineResult",
    "RuleResult",
    "TupleEvaluation",
    "TupleEvaluator",
    "TupleNormalizer",
    "ValidationPipeline",
    "ValidationReport",
    "ValidationSeverity",
    "evaluate_alignment",
    "validate_matches",
]