"""Implement the inference of the schema a graph follows today, as a draft to review."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field

from kgeval.graph.model import PropertyTypeName, PropertyValue, value_type
from kgeval.graph.view import GraphView
from kgeval.schema.model import (
    Constraint,
    GraphSchema,
    NodeType,
    Pattern,
    PropertyType,
    RelationshipType,
)

__all__ = ["infer_schema"]


@dataclass
class _Usage:
    """Tally how often a type occurs and which value types its properties hold."""

    count: int = 0
    property_types: defaultdict[str, Counter[PropertyTypeName]] = field(
        default_factory=lambda: defaultdict(Counter)
    )

    def add(self, properties: Mapping[str, PropertyValue]) -> None:
        self.count += 1
        for key, value in properties.items():
            self.property_types[key][value_type(value)] += 1

    def declared_properties(self) -> tuple[PropertyType, ...]:
        # Type each property by its most common value type, ties going to the alphabetically first.
        return tuple(
            PropertyType(name=key, type=min(types, key=lambda name: (-types[name], name)))
            for key, types in sorted(self.property_types.items())
        )

    def universal_properties(self) -> list[str]:
        return sorted(
            key for key, types in self.property_types.items() if types.total() == self.count
        )


def infer_schema(graph: GraphView) -> GraphSchema:
    """Return the graph schema a graph's entities and facts follow today.

    The result is a starting point to review and tighten, not a statement of what is
    correct. Every node type of an entity and every relationship type of a fact is
    declared, ordered from most to least used, with every property typed by its most
    common value type. Every observed (source node type, relationship type, target node
    type) combination becomes a pattern, also ordered by use, so rare patterns, which
    are often extraction mistakes, come last. Every property that all entities of a node
    type, or all facts of a relationship type, have gets an EXISTENCE constraint.
    Entities without a node type are left out.

    Args:
        graph: The graph to describe.

    Returns:
        The inferred graph schema, closed to additional properties, and to additional
        node types, relationship types, and patterns when it declares some.
    """
    node_usage: defaultdict[str, _Usage] = defaultdict(_Usage)
    relationship_usage: defaultdict[str, _Usage] = defaultdict(_Usage)
    pattern_counts: Counter[tuple[str, str, str]] = Counter()
    for entity in graph.entities:
        for label in graph.layout.node_types(entity):
            node_usage[label].add(entity.properties)
    for fact in graph.facts:
        relationship_type = graph.relationship_type(fact)
        relationship_usage[relationship_type].add(fact.properties)
        for source in graph.layout.node_types(graph.node(fact.source_id)):
            for target in graph.layout.node_types(graph.node(fact.target_id)):
                pattern_counts[(source, relationship_type, target)] += 1

    node_types = _by_frequency(node_usage)
    relationship_types = _by_frequency(relationship_usage)
    patterns = sorted(pattern_counts.items(), key=lambda item: (-item[1], item[0]))
    return GraphSchema(
        node_types=tuple(
            NodeType(
                label=label, properties=usage.declared_properties(), additional_properties=False
            )
            for label, usage in node_types
        ),
        relationship_types=tuple(
            RelationshipType(
                label=label, properties=usage.declared_properties(), additional_properties=False
            )
            for label, usage in relationship_types
        ),
        patterns=tuple(
            Pattern(source=source, relationship=relationship, target=target)
            for (source, relationship, target), _ in patterns
        ),
        constraints=(
            *(
                Constraint(type="EXISTENCE", property_names=(key,), node_type=label)
                for label, usage in node_types
                for key in usage.universal_properties()
            ),
            *(
                Constraint(type="EXISTENCE", property_names=(key,), relationship_type=label)
                for label, usage in relationship_types
                for key in usage.universal_properties()
            ),
        ),
        # Close only the lists that declare something, since a closed empty list rejects everything.
        additional_node_types=not node_types,
        additional_relationship_types=not relationship_types,
        additional_patterns=not patterns,
    )


def _by_frequency(usages: Mapping[str, _Usage]) -> list[tuple[str, _Usage]]:
    return sorted(usages.items(), key=lambda item: (-item[1].count, item[0]))
