"""Define the base class of the schema metrics."""

from __future__ import annotations

from typing import ClassVar

from kgeval.metrics.base import InputName, Inputs, Metric, MetricCategory

__all__ = ["SchemaMetric"]


class SchemaMetric(Metric):
    """Measure how a graph matches the graph schema among the inputs.

    The graph schema's digest is part of every signature, so results against different
    graph schemas are never mistaken for each other.
    """

    category: ClassVar[MetricCategory] = MetricCategory.SCHEMA
    needs: ClassVar[tuple[InputName, ...]] = ("schema",)

    def signature(self, inputs: Inputs) -> str:
        """Return the metric name with a digest of the schema.

        Args:
            inputs: The inputs the metric is evaluated with.

        Returns:
            The signature.
        """
        return f"{self.name}|schema={self.require(inputs.schema, 'schema').digest()}"
