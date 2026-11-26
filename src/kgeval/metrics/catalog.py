"""Implement the catalog of built-in metrics by category, and the loading of suites."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, TypeAdapter, ValidationError

from kgeval._yaml import load_yaml
from kgeval.errors import InputError
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
from kgeval.metrics.model import (
    EntityDisambiguation,
    EntityResolution,
    ExtractionCompleteness,
    FactAccuracy,
    ModelMetric,
    NodeTypeAccuracy,
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

__all__ = [
    "BUILTIN_METRICS",
    "GROUND_TRUTH_METRICS",
    "INTRINSIC_METRICS",
    "MODEL_METRICS",
    "REFERENCE_METRICS",
    "RETRIEVAL_METRICS",
    "SCHEMA_METRICS",
    "load_suite",
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

MODEL_METRICS: tuple[type[ModelMetric], ...] = (
    EntityResolution,
    EntityDisambiguation,
    NodeTypeAccuracy,
    FactAccuracy,
    VocabularyResolution,
    ExtractionCompleteness,
)
"""The metrics that need a model, in report order."""

RETRIEVAL_METRICS: tuple[type[RetrievalMetric], ...] = (RetrievalRecall, RetrievalPrecision)
"""The metrics that evaluate retrieval from the graph, in report order."""

BUILTIN_METRICS: tuple[type[Metric], ...] = (
    *INTRINSIC_METRICS,
    *SCHEMA_METRICS,
    *REFERENCE_METRICS,
    *MODEL_METRICS,
    *RETRIEVAL_METRICS,
)
"""Every built-in metric, grouped by category."""


_Suite = Annotated[dict[str, dict[str, Any] | None], Field(min_length=1)]


def load_suite(path: str | Path) -> list[Metric]:
    """Load the metrics that a suite file lists, with their settings.

    A suite maps built-in metric names to settings, which are the metrics' parameters,
    such as ``threshold``, ``sample_size``, or ``k``; a metric without settings needs
    only its name. A suite holds no inputs, so one suite serves every graph and model.

    Args:
        path: The suite file, YAML or JSON.

    Returns:
        The metrics, in the order the suite lists them.

    Raises:
        InputError: If the file cannot be read, names an unknown metric, or holds an
            invalid setting.
    """
    suite = load_yaml(path, TypeAdapter(_Suite))
    metric_types = {metric_type.name: metric_type for metric_type in BUILTIN_METRICS}
    metrics: list[Metric] = []
    for name, settings in suite.items():
        if name not in metric_types:
            matches = difflib.get_close_matches(name, metric_types, n=1)
            hint = f"; did you mean {matches[0]}?" if matches else ""
            raise InputError(f"invalid suite {path}: unknown metric {name}{hint}")
        try:
            metrics.append(metric_types[name].model_validate(settings or {}))
        except ValidationError as error:
            raise InputError(f"invalid suite {path}: {name}: {error}") from error
    return metrics
