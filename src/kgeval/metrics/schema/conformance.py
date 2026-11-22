"""Implement the entity conformance and fact conformance metrics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

from kgeval.graph.model import Node, Relationship, canonical_json, property_types
from kgeval.graph.view import GraphView
from kgeval.metrics.base import Finding, MetricResult
from kgeval.metrics.schema.base import SchemaMetric
from kgeval.schema.model import Constraint, GraphSchema, NodeType, RelationshipType

__all__ = ["EntityConformance", "FactConformance"]


class EntityConformance(SchemaMetric):
    """Measure the share of entities that conform to the graph schema.

    An entity conforms when all of these hold:

    - unless the graph schema allows additional node types, it has a node type and every
      node type it has is declared;
    - every declared property it has holds a value of the declared type, and every
      required one is present;
    - unless one of its declared node types allows additional properties, it has no
      undeclared property;
    - it satisfies every EXISTENCE, UNIQUENESS, and KEY constraint on its node types.

    Lexical nodes and community nodes are not evaluated: as in neo4j-graphrag, graph
    schemas describe only extracted entities and the relationships between them.
    """

    name: ClassVar[str] = "entity_conformance"
    unit: ClassVar[str] = "entities"

    def evaluate(self, graph: GraphView) -> MetricResult:
        """Check every entity against the graph schema and report each that fails.

        Args:
            graph: The graph to evaluate.

        Returns:
            The metric's result.
        """
        index = _SchemaIndex.of(self.graph_schema)
        problems: defaultdict[str, list[str]] = defaultdict(list)
        for entity in graph.entities:
            problems[entity.id] += index.entity_problems(entity, graph.layout.node_types(entity))
        for constraint in self.graph_schema.constraints:
            if constraint.node_type is not None:
                for element_id, problem in _constraint_problems(graph, constraint):
                    problems[element_id].append(problem)
        findings = _findings(graph, graph.entities, problems)
        return self.result(len(graph.entities), len(findings), findings)


class FactConformance(SchemaMetric):
    """Measure the share of facts that conform to the graph schema.

    A fact conforms when all of these hold:

    - unless the graph schema allows additional relationship types, its relationship
      type is declared;
    - unless the graph schema allows additional patterns, one combination of a node
      type of its source, its relationship type, and a node type of its target is a
      declared pattern;
    - its properties follow its declared relationship type, as for entities;
    - it satisfies every EXISTENCE, UNIQUENESS, and KEY constraint on its relationship
      type;
    - when its relationship type declares ``max_targets``, its source has facts of that
      type to no more distinct targets: two birthplaces contradict each other, so every
      fact of the type from that source fails.

    The pattern rule alone is Text2KGBench's ontology conformance, the share of facts
    that conform to the ontology; this metric adds property, constraint, and cardinality
    checks.
    """

    name: ClassVar[str] = "fact_conformance"
    unit: ClassVar[str] = "facts"

    def evaluate(self, graph: GraphView) -> MetricResult:
        """Check every fact against the graph schema and report each that fails.

        Args:
            graph: The graph to evaluate.

        Returns:
            The metric's result.
        """
        index = _SchemaIndex.of(self.graph_schema)
        problems: defaultdict[str, list[str]] = defaultdict(list)
        for fact in graph.facts:
            source_types = graph.layout.node_types(graph.node(fact.source_id))
            target_types = graph.layout.node_types(graph.node(fact.target_id))
            problems[fact.id] += index.fact_problems(
                fact, graph.relationship_type(fact), source_types, target_types
            )
        for constraint in self.graph_schema.constraints:
            if constraint.relationship_type is not None:
                for element_id, problem in _constraint_problems(graph, constraint):
                    problems[element_id].append(problem)
        for element_id, problem in _cardinality_problems(graph, self.graph_schema):
            problems[element_id].append(problem)
        findings = _findings(graph, graph.facts, problems)
        return self.result(len(graph.facts), len(findings), findings)


@dataclass(frozen=True)
class _SchemaIndex:
    """Look up a graph schema's node types, relationship types, and patterns in constant time."""

    schema: GraphSchema
    node_types: dict[str, NodeType]
    relationship_types: dict[str, RelationshipType]
    patterns: frozenset[tuple[str, str, str]]

    @classmethod
    def of(cls, schema: GraphSchema) -> _SchemaIndex:
        return cls(
            schema=schema,
            node_types={node_type.label: node_type for node_type in schema.node_types},
            relationship_types={
                relationship.label: relationship for relationship in schema.relationship_types
            },
            patterns=frozenset(
                (pattern.source, pattern.relationship, pattern.target)
                for pattern in schema.patterns
            ),
        )

    def entity_problems(self, entity: Node, node_types: tuple[str, ...]) -> list[str]:
        problems: list[str] = []
        if not self.schema.additional_node_types:
            if not node_types:
                problems.append("no node type")
            undeclared = [node_type for node_type in node_types if node_type not in self.node_types]
            if undeclared:
                noun = "node type" if len(undeclared) == 1 else "node types"
                problems.append(f"undeclared {noun} {', '.join(undeclared)}")
        declared = [
            self.node_types[node_type] for node_type in node_types if node_type in self.node_types
        ]
        return problems + _property_problems(entity, declared)

    def fact_problems(
        self,
        fact: Relationship,
        relationship_type: str,
        source_types: tuple[str, ...],
        target_types: tuple[str, ...],
    ) -> list[str]:
        problems: list[str] = []
        declared = self.relationship_types.get(relationship_type)
        if declared is None and not self.schema.additional_relationship_types:
            # Skip the pattern check: no declared pattern can name an undeclared type.
            problems.append(f"undeclared relationship type {relationship_type}")
        elif not self.schema.additional_patterns and not any(
            (source, relationship_type, target) in self.patterns
            for source in source_types
            for target in target_types
        ):
            pattern = (
                f"({':'.join(source_types)})-[:{relationship_type}]->({':'.join(target_types)})"
            )
            problems.append(f"pattern {pattern} is not declared")
        return problems + _property_problems(fact, [declared] if declared else [])


def _property_problems(
    element: Node | Relationship, types: Sequence[NodeType | RelationshipType]
) -> list[str]:
    if not types:
        return []
    declared = {
        property_type.name: property_type
        for element_type in types
        for property_type in element_type.properties
    }
    held = property_types(element)
    problems: list[str] = []
    for property_type in declared.values():
        if property_type.name in held:
            actual = held[property_type.name]
            if actual != property_type.type:
                problems.append(
                    f"property {property_type.name} is {actual}, expected {property_type.type}"
                )
        elif property_type.required:
            problems.append(f"required property {property_type.name} is missing")
    if not any(element_type.additional_properties for element_type in types):
        undeclared = sorted(key for key in element.properties if key not in declared)
        if undeclared:
            noun = "property" if len(undeclared) == 1 else "properties"
            problems.append(f"undeclared {noun} {', '.join(undeclared)}")
    return problems


def _constraint_problems(graph: GraphView, constraint: Constraint) -> Iterator[tuple[str, str]]:
    """Yield the ID and a description of every element that breaks a constraint."""
    elements: list[Node | Relationship]
    if constraint.node_type is not None:
        scope = constraint.node_type
        elements = [entity for entity in graph.entities if scope in graph.layout.node_types(entity)]
    else:
        # Rely on validation, which requires a relationship type when no node type is set.
        scope = str(constraint.relationship_type)
        elements = [fact for fact in graph.facts if graph.relationship_type(fact) == scope]
    names = constraint.property_names
    if constraint.type in ("EXISTENCE", "KEY"):
        for element in elements:
            for missing in (key for key in names if key not in property_types(element)):
                yield (
                    element.id,
                    f"{constraint.type} constraint on {scope} needs property {missing}",
                )
    if constraint.type in ("UNIQUENESS", "KEY"):
        for element_id, others in _collisions(elements, names).items():
            yield (
                element_id,
                f"{constraint.type} constraint on {scope} needs unique {', '.join(names)}, "
                f"shared with {others} other {'element' if others == 1 else 'elements'}",
            )


def _cardinality_problems(graph: GraphView, schema: GraphSchema) -> Iterator[tuple[str, str]]:
    """Yield the ID and a description of every fact whose source has too many targets."""
    limits = {
        relationship_type.label: relationship_type.max_targets
        for relationship_type in schema.relationship_types
        if relationship_type.max_targets is not None
    }
    facts: defaultdict[tuple[str, str], list[Relationship]] = defaultdict(list)
    for fact in graph.facts:
        if (relationship_type := graph.relationship_type(fact)) in limits:
            facts[(fact.source_id, relationship_type)].append(fact)
    for (_, relationship_type), group in facts.items():
        limit = limits[relationship_type]
        targets = len({fact.target_id for fact in group})
        if targets > limit:
            for fact in group:
                yield (
                    fact.id,
                    f"{relationship_type} allows {limit} {'target' if limit == 1 else 'targets'} "
                    f"per source, and its source has {targets}",
                )


def _collisions(elements: Sequence[Node | Relationship], names: tuple[str, ...]) -> dict[str, int]:
    """Map each element whose constrained values another element shares to how many do."""
    groups: defaultdict[str, list[str]] = defaultdict(list)
    for element in elements:
        if all(key in element.properties for key in names):
            values = canonical_json([element.properties[key] for key in names])
            groups[values].append(element.id)
    return {
        element_id: len(ids) - 1 for ids in groups.values() if len(ids) > 1 for element_id in ids
    }


def _findings(
    graph: GraphView, elements: Iterable[Node | Relationship], problems: Mapping[str, list[str]]
) -> list[Finding]:
    findings: list[Finding] = []
    for element in elements:
        reasons = list(dict.fromkeys(problems.get(element.id, ())))
        if not reasons:
            continue
        description = graph.describe(element)
        findings.append(
            Finding.about(element, f"{description}: {'; '.join(reasons)}", subject=description)
        )
    return findings
