"""Implement the property type consistency and embedding consistency metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import ClassVar, TypeAlias

from kgeval.graph.layout import Role
from kgeval.graph.model import Node, PropertyTypeName, Relationship, value_type
from kgeval.graph.view import GraphView
from kgeval.metrics.base import Finding, Inputs, Metric, MetricCategory, MetricResult

__all__ = ["EmbeddingConsistency", "PropertyTypeConsistency"]

_Element: TypeAlias = Node | Relationship


class PropertyTypeConsistency(Metric):
    """Measure the share of property values whose type matches the rest of their group.

    Values are grouped by property key and by an entity's node types, taken together, or
    a fact's relationship type. In each group the most common value type counts as
    intended, ties going to the alphabetically first type name, and every other value
    fails. Only entities and facts are evaluated.
    """

    name: ClassVar[str] = "property_type_consistency"
    category: ClassVar[MetricCategory] = MetricCategory.INTRINSIC
    unit: ClassVar[str] = "property values"

    def evaluate(self, graph: GraphView, inputs: Inputs) -> MetricResult:
        """Report every property value whose type differs from the one its group holds.

        Args:
            graph: The graph to evaluate.
            inputs: The inputs the metric is evaluated with.

        Returns:
            The metric's result.
        """
        groups: defaultdict[tuple[str, str], defaultdict[PropertyTypeName, list[_Element]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        elements: tuple[_Element, ...] = (*graph.entities, *graph.facts)
        for element in elements:
            group = _group(graph, element)
            for key, value in element.properties.items():
                groups[(group, key)][value_type(value)].append(element)
        findings: list[Finding] = []
        for (group, key), by_type in sorted(groups.items()):
            intended = min(by_type, key=lambda type_name: (-len(by_type[type_name]), type_name))
            findings += [
                Finding.about(
                    element,
                    f"{graph.describe(element)}: property {key} is {type_name}, "
                    f"while other {group} hold {intended}",
                    subject=f"{graph.describe(element)}: property {key}",
                )
                for type_name, elements in sorted(by_type.items())
                if type_name != intended
                for element in elements
            ]
        evaluated = sum(
            len(elements) for by_type in groups.values() for elements in by_type.values()
        )
        return self.result(inputs, evaluated, len(findings), findings)


class EmbeddingConsistency(Metric):
    """Measure the share of expected embeddings that elements hold, with the expected dimension.

    Vector search reaches a graph through its embeddings, so an element whose embedding
    is missing, or has another dimension than the vector index expects, cannot be found.
    Entities are grouped by node types, taken together, lexical and community nodes by
    labels, and facts by relationship type. As for property types, what most members
    hold is intended, so a group expects every embedding property that at least half of
    its members hold, with the dimension most of those members have, ties going to the
    smaller dimension, so a stray embedding on a few members fails no one. Each member
    of the group is one unit per expected embedding, and fails when it lacks the
    embedding or holds another dimension. kgeval reads every non-empty list of floats as
    an embedding.
    """

    name: ClassVar[str] = "embedding_consistency"
    category: ClassVar[MetricCategory] = MetricCategory.INTRINSIC
    unit: ClassVar[str] = "expected embeddings"

    def evaluate(self, graph: GraphView, inputs: Inputs) -> MetricResult:
        """Report every expected embedding that is missing or has another dimension.

        Args:
            graph: The graph to evaluate.
            inputs: The inputs the metric is evaluated with.

        Returns:
            The metric's result.
        """
        elements: tuple[_Element, ...] = (
            *graph.entities,
            *graph.lexical_nodes,
            *graph.community_nodes,
            *graph.facts,
        )
        groups: defaultdict[str, list[_Element]] = defaultdict(list)
        for element in elements:
            groups[_group(graph, element)].append(element)
        evaluated = 0
        findings: list[Finding] = []
        for group, members in sorted(groups.items()):
            dimensions: defaultdict[str, Counter[int]] = defaultdict(Counter)
            for member in members:
                for key, dimension in member.embedding_dimensions.items():
                    dimensions[key][dimension] += 1
            for key, counts in sorted(dimensions.items()):
                if 2 * counts.total() < len(members):
                    continue
                expected = min(counts, key=lambda dimension: (-counts[dimension], dimension))
                evaluated += len(members)
                for member in members:
                    actual = member.embedding_dimensions.get(key)
                    if actual == expected:
                        continue
                    if actual is None:
                        problem = f"no embedding under property {key}, while other {group} have one"
                    else:
                        problem = (
                            f"the embedding under property {key} has {actual} dimensions, "
                            f"while other {group} have {expected}"
                        )
                    findings.append(
                        Finding.about(
                            member,
                            f"{graph.describe(member)}: {problem}",
                            subject=f"{graph.describe(member)}: embedding {key}",
                        )
                    )
        if not evaluated:
            return self.skip(
                inputs,
                "no group holds an embedding on at least half its members; kgeval reads "
                "non-empty lists of floats as embeddings",
            )
        return self.result(inputs, evaluated, len(findings), findings)


def _group(graph: GraphView, element: _Element) -> str:
    """Name the group of elements that an element is compared with."""
    if isinstance(element, Relationship):
        return f"{graph.relationship_type(element)} facts"
    if graph.role(element) is Role.ENTITY:
        node_types = graph.layout.node_types(element)
        return f"{':'.join(node_types)} entities" if node_types else "entities without a node type"
    return f"{':'.join(element.labels)} nodes"
