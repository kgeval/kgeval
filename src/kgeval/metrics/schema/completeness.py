"""Implement the schema completeness and property completeness metrics."""

from __future__ import annotations

from typing import ClassVar

from kgeval.graph.model import Node, Relationship, property_types
from kgeval.graph.view import GraphView
from kgeval.metrics.base import Finding, Inputs, MetricResult
from kgeval.metrics.schema.base import SchemaMetric
from kgeval.schema.model import Pattern

__all__ = ["PropertyCompleteness", "SchemaCompleteness"]


class SchemaCompleteness(SchemaMetric):
    """Measure the share of declared node types, relationship types, and patterns in use.

    A node type is in use when an entity has it, a relationship type when a fact has it,
    and a pattern when a fact realizes it by having its relationship type, a source with
    its source node type, and a target with its target node type. This is the schema
    completeness of the linked data quality literature; for node types alone, it is the
    class richness of OntoQA.
    """

    name: ClassVar[str] = "schema_completeness"
    unit: ClassVar[str] = "declared types and patterns"

    def evaluate(self, graph: GraphView, inputs: Inputs) -> MetricResult:
        """Report every declared node type, relationship type, and pattern the graph lacks.

        Args:
            graph: The graph to evaluate.
            inputs: The inputs the metric is evaluated with.

        Returns:
            The metric's result.

        Raises:
            InputError: If no graph schema was given.
        """
        schema = self.require(inputs.schema, "schema")
        node_types = {
            label for entity in graph.entities for label in graph.layout.node_types(entity)
        }
        relationship_types = {graph.relationship_type(fact) for fact in graph.facts}
        patterns = {
            (source_type, graph.relationship_type(fact), target_type)
            for fact in graph.facts
            for source_type in graph.layout.node_types(graph.node(fact.source_id))
            for target_type in graph.layout.node_types(graph.node(fact.target_id))
        }
        missing = [
            (f"node type {node_type.label}", "no entity has the declared")
            for node_type in schema.node_types
            if node_type.label not in node_types
        ]
        missing += [
            (f"relationship type {relationship_type.label}", "no fact has the declared")
            for relationship_type in schema.relationship_types
            if relationship_type.label not in relationship_types
        ]
        missing += [
            (f"pattern {_describe(pattern)}", "no fact realizes the declared")
            for pattern in schema.patterns
            if (pattern.source, pattern.relationship, pattern.target) not in patterns
        ]
        evaluated = len(schema.node_types) + len(schema.relationship_types) + len(schema.patterns)
        findings = [
            Finding(message=f"{problem} {subject}", subject=subject) for subject, problem in missing
        ]
        return self.result(inputs, evaluated, len(findings), findings)


class PropertyCompleteness(SchemaMetric):
    """Measure the share of declared properties that entities and facts actually hold.

    Each property declared by any of an entity's node types is one unit for that entity,
    and each property declared by a fact's relationship type is one unit for that fact.
    A unit fails when the entity or fact lacks the property. This is the property, or
    column, completeness of the data quality literature, which measures how filled-in
    the graph is, not whether it is wrong, so legitimately optional properties lower the
    score without being defects.
    """

    name: ClassVar[str] = "property_completeness"
    unit: ClassVar[str] = "declared properties"

    def evaluate(self, graph: GraphView, inputs: Inputs) -> MetricResult:
        """Report every declared property that an entity or fact lacks.

        Args:
            graph: The graph to evaluate.
            inputs: The inputs the metric is evaluated with.

        Returns:
            The metric's result.

        Raises:
            InputError: If no graph schema was given.
        """
        schema = self.require(inputs.schema, "schema")
        node_types = {node_type.label: node_type for node_type in schema.node_types}
        relationship_types = {
            relationship_type.label: relationship_type
            for relationship_type in schema.relationship_types
        }
        units: list[tuple[Node | Relationship, list[str]]] = [
            (
                entity,
                sorted(
                    {
                        declared.name
                        for node_type in graph.layout.node_types(entity)
                        if node_type in node_types
                        for declared in node_types[node_type].properties
                    }
                ),
            )
            for entity in graph.entities
        ]
        units += [
            (
                fact,
                sorted(
                    declared.name for declared in relationship_types[relationship_type].properties
                ),
            )
            for fact in graph.facts
            if (relationship_type := graph.relationship_type(fact)) in relationship_types
        ]
        findings = [
            Finding.about(
                element,
                f"{graph.describe(element)}: declared property {name} is missing",
                subject=f"{graph.describe(element)}: property {name}",
            )
            for element, names in units
            for name in names
            if name not in property_types(element)
        ]
        return self.result(inputs, sum(len(names) for _, names in units), len(findings), findings)


def _describe(pattern: Pattern) -> str:
    return f"({pattern.source})-[:{pattern.relationship}]->({pattern.target})"
