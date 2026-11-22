"""Provide the schema metrics."""

from kgeval.metrics.schema.base import SchemaMetric
from kgeval.metrics.schema.completeness import PropertyCompleteness, SchemaCompleteness
from kgeval.metrics.schema.conformance import EntityConformance, FactConformance

__all__ = [
    "EntityConformance",
    "FactConformance",
    "PropertyCompleteness",
    "SchemaCompleteness",
    "SchemaMetric",
]
