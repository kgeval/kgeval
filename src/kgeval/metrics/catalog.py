"""List the built-in metrics by category, and assemble the suite that inputs enable."""

from __future__ import annotations

from kgeval.errors import InputError
from kgeval.ground_truth import GroundTruth
from kgeval.judges.base import Judge
from kgeval.metrics.base import Metric
from kgeval.metrics.intrinsic import (
    EmbeddingConsistency,
    EntityConnectivity,
    EntityNaming,
    EntityProvenance,
    EntityUniqueness,
    ExtractionCoverage,
    FactProvenance,
    FactUniqueness,
    PropertyTypeConsistency,
    VocabularyUniqueness,
)
from kgeval.metrics.judged import (
    DEFAULT_SAMPLE_SIZE,
    EntityDisambiguation,
    EntityResolution,
    ExtractionCompleteness,
    FactAccuracy,
    JudgedMetric,
    NodeTypeAccuracy,
    SampledMetric,
    VocabularyResolution,
)
from kgeval.metrics.reference import (
    Competency,
    EntityF1,
    EntityPrecision,
    EntityRecall,
    FactF1,
    FactPrecision,
    FactRecall,
    GroundTruthMetric,
)
from kgeval.metrics.retrieval import RetrievalMetric, RetrievalPrecision, RetrievalRecall
from kgeval.metrics.schema import (
    EntityConformance,
    FactConformance,
    PropertyCompleteness,
    SchemaCompleteness,
    SchemaMetric,
)
from kgeval.questions import QuestionSet
from kgeval.retrieval import Retrievals
from kgeval.schema.model import GraphSchema

__all__ = [
    "BUILTIN_METRICS",
    "GROUND_TRUTH_METRICS",
    "INTRINSIC_METRICS",
    "JUDGED_METRICS",
    "REFERENCE_METRICS",
    "RETRIEVAL_METRICS",
    "SCHEMA_METRICS",
    "default_metrics",
]

INTRINSIC_METRICS: tuple[type[Metric], ...] = (
    EntityUniqueness,
    EntityConnectivity,
    EntityNaming,
    EntityProvenance,
    FactUniqueness,
    FactProvenance,
    VocabularyUniqueness,
    PropertyTypeConsistency,
    EmbeddingConsistency,
    ExtractionCoverage,
)
"""The deterministic metrics that need nothing but the graph, in report order."""

SCHEMA_METRICS: tuple[type[SchemaMetric], ...] = (
    EntityConformance,
    FactConformance,
    SchemaCompleteness,
    PropertyCompleteness,
)
"""The deterministic metrics that measure a graph against a graph schema, in report order."""

GROUND_TRUTH_METRICS: tuple[type[GroundTruthMetric], ...] = (
    FactPrecision,
    FactRecall,
    FactF1,
    EntityPrecision,
    EntityRecall,
    EntityF1,
)
"""The metrics that compare the graph with a ground truth, in report order."""

REFERENCE_METRICS: tuple[type[Metric], ...] = (*GROUND_TRUTH_METRICS, Competency)
"""The ground truth metrics and competency, which need a reference, in report order."""

JUDGED_METRICS: tuple[type[JudgedMetric], ...] = (
    EntityResolution,
    EntityDisambiguation,
    NodeTypeAccuracy,
    FactAccuracy,
    VocabularyResolution,
    ExtractionCompleteness,
)
"""The metrics that need an LLM judge, in report order."""

RETRIEVAL_METRICS: tuple[type[RetrievalMetric], ...] = (RetrievalRecall, RetrievalPrecision)
"""The metrics that evaluate retrieval from the graph, in report order."""

BUILTIN_METRICS: tuple[type[Metric], ...] = (
    *INTRINSIC_METRICS,
    *SCHEMA_METRICS,
    *REFERENCE_METRICS,
    *JUDGED_METRICS,
    *RETRIEVAL_METRICS,
)
"""Every built-in metric, grouped by category."""


def default_metrics(
    *,
    schema: GraphSchema | None = None,
    ground_truth: GroundTruth | None = None,
    questions: QuestionSet | None = None,
    retrievals: Retrievals | None = None,
    judge: Judge | None = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = 0,
    top_k: int | None = None,
) -> list[Metric]:
    """Return every built-in metric that the given inputs enable, without thresholds.

    The intrinsic metrics need nothing and are always included; each input adds the
    metrics that need it.

    Args:
        schema: A graph schema; adds the schema metrics.
        ground_truth: A ground truth; adds the ground truth metrics.
        questions: Test questions; adds ``competency``.
        retrievals: What a retriever returned for each of the questions; adds the
            retrieval metrics.
        judge: An LLM judge; adds the judged metrics.
        sample_size: The most units each sampled judged metric judges.
        seed: Selects which units the judged metrics sample.
        top_k: How many top-ranked element IDs of each retrieval the retrieval metrics
            evaluate; all of them when omitted.

    Returns:
        The metrics, grouped by category.

    Raises:
        InputError: If retrievals are given without the questions they answer.
    """
    metrics: list[Metric] = [metric_type() for metric_type in INTRINSIC_METRICS]
    if schema is not None:
        metrics += [metric_type(graph_schema=schema) for metric_type in SCHEMA_METRICS]
    if ground_truth is not None:
        metrics += [metric_type(ground_truth=ground_truth) for metric_type in GROUND_TRUTH_METRICS]
    if questions is not None:
        metrics.append(Competency(questions=questions))
    if judge is not None:
        metrics += [
            metric_type(judge=judge, sample_size=sample_size, seed=seed)
            if issubclass(metric_type, SampledMetric)
            else metric_type(judge=judge)
            for metric_type in JUDGED_METRICS
        ]
    if retrievals is not None:
        if questions is None:
            raise InputError("retrievals need the test questions they answer")
        metrics += [
            metric_type(questions=questions, retrievals=retrievals, k=top_k)
            for metric_type in RETRIEVAL_METRICS
        ]
    return metrics
