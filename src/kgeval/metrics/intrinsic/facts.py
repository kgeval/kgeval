"""Implement the fact uniqueness metric."""

from __future__ import annotations

from collections import defaultdict
from typing import ClassVar

from kgeval.graph.model import Relationship
from kgeval.graph.view import GraphView
from kgeval.metrics.base import Finding, Inputs, Metric, MetricCategory, MetricResult

__all__ = ["FactUniqueness"]


class FactUniqueness(Metric):
    """Measure the share of facts that are distinct, counting repeated facts once.

    Facts repeat each other when they share source node, relationship type, and target
    node, as ``GraphView.relationship_type`` reads the type, whatever their properties,
    since a fact is its source, relationship type, and target, as the ground truth
    metrics also match it, and its properties describe it. Repeats come from running a
    ``CREATE``-based import twice, or from storing every extraction of a fact as its own
    relationship, with its own ID, timestamp, or chunk. ``failed`` counts redundant
    copies, so n repeats of a fact add n - 1.
    """

    name: ClassVar[str] = "fact_uniqueness"
    category: ClassVar[MetricCategory] = MetricCategory.INTRINSIC
    unit: ClassVar[str] = "facts"

    def evaluate(self, graph: GraphView, inputs: Inputs) -> MetricResult:
        """Group the facts that repeat each other and report every group of two or more.

        Args:
            graph: The graph to evaluate.
            inputs: The inputs the metric is evaluated with.

        Returns:
            The metric's result.
        """
        groups: defaultdict[tuple[str, str, str], list[Relationship]] = defaultdict(list)
        for fact in graph.facts:
            groups[(fact.source_id, graph.relationship_type(fact), fact.target_id)].append(fact)
        duplicates = [group for group in groups.values() if len(group) > 1]
        findings = [
            Finding.about(
                group,
                f"{len(group)} facts repeat {graph.describe(group[0])}",
                subject=graph.describe(group[0]),
            )
            for group in duplicates
        ]
        failed = sum(len(group) - 1 for group in duplicates)
        return self.result(inputs, len(graph.facts), failed, findings)
