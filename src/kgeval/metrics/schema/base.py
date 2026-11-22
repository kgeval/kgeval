"""Define the base class of the schema metrics."""

from __future__ import annotations

from typing import ClassVar

from kgeval.metrics.base import Metric, MetricCategory
from kgeval.schema.model import GraphSchema

__all__ = ["SchemaMetric"]


class SchemaMetric(Metric):
    """Measure how a graph matches a graph schema.

    The graph schema's digest is part of every signature, so results against different
    graph schemas are never mistaken for each other.

    Attributes:
        graph_schema: The graph schema to measure against.
    """

    category: ClassVar[MetricCategory] = MetricCategory.SCHEMA

    graph_schema: GraphSchema

    def signature(self) -> str:
        """Return the metric name with a digest of the schema.

        Returns:
            The signature.
        """
        return f"{self.name}|schema={self.graph_schema.digest()}"
