"""Implement the entity uniqueness, connectivity, and naming metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import ClassVar

from kgeval.graph.model import Node
from kgeval.graph.names import normalize_name
from kgeval.graph.view import GraphView
from kgeval.metrics.base import Finding, Metric, MetricCategory, MetricResult

__all__ = ["EntityConnectivity", "EntityNaming", "EntityUniqueness"]

_PRONOUNS = frozenset(
    {
        "i", "me", "my", "mine", "myself",
        "you", "your", "yours", "yourself", "yourselves",
        "he", "him", "his", "himself",
        "she", "her", "hers", "herself",
        "it", "its", "itself",
        "we", "us", "our", "ours", "ourselves",
        "they", "them", "their", "theirs", "themselves",
        "this", "that", "these", "those",
        "who", "whom", "whose", "which", "what",
        "someone", "somebody", "anyone", "anybody", "everyone", "everybody", "no one", "nobody",
    }
)  # fmt: skip


class EntityUniqueness(Metric):
    """Measure the share of entities with a name that are distinct, counting duplicates once.

    Entities are duplicates when their names share the key that ``normalize_name``
    returns, whatever their node types; findings list each entity's node types so that
    homonyms can be dismissed. Entities without a name that has a letter or digit are
    not evaluated. ``failed`` counts redundant entities: a group of n duplicates adds
    n - 1, the merges that entity resolution would make.
    """

    name: ClassVar[str] = "entity_uniqueness"
    category: ClassVar[MetricCategory] = MetricCategory.INTRINSIC
    unit: ClassVar[str] = "entities with a name"

    def evaluate(self, graph: GraphView) -> MetricResult:
        """Group entities by normalized name and report every group of two or more.

        Args:
            graph: The graph to evaluate.

        Returns:
            The metric's result.
        """
        groups: defaultdict[str, list[Node]] = defaultdict(list)
        for entity in graph.entities:
            key = normalize_name(graph.name(entity) or "")
            if key:
                groups[key].append(entity)
        duplicates = sorted(
            (item for item in groups.items() if len(item[1]) > 1), key=lambda item: item[0]
        )
        findings = [
            Finding(
                message=f"{len(group)} entities share the normalized name {key!r}: "
                + ", ".join(sorted(graph.describe(entity) for entity in group)),
                subject=key,
                node_ids=tuple(entity.id for entity in group),
            )
            for key, group in duplicates
        ]
        evaluated = sum(len(group) for group in groups.values())
        failed = sum(len(group) - 1 for _, group in duplicates)
        return self.result(evaluated, failed, findings)


class EntityConnectivity(Metric):
    """Measure the share of entities that are the source or target of a fact with another entity.

    An isolated entity, in no such fact, cannot be reached by a traversal from any other
    entity. A self-loop does not count.
    """

    name: ClassVar[str] = "entity_connectivity"
    category: ClassVar[MetricCategory] = MetricCategory.INTRINSIC
    unit: ClassVar[str] = "entities"

    def evaluate(self, graph: GraphView) -> MetricResult:
        """Report every isolated entity.

        Args:
            graph: The graph to evaluate.

        Returns:
            The metric's result.
        """
        connected = {
            node_id
            for fact in graph.facts
            if fact.source_id != fact.target_id
            for node_id in (fact.source_id, fact.target_id)
        }
        findings = [
            Finding(
                message=f"{graph.describe(entity)} is in no fact with another entity",
                subject=graph.describe(entity),
                node_ids=(entity.id,),
            )
            for entity in graph.entities
            if entity.id not in connected
        ]
        return self.result(len(graph.entities), len(findings), findings)


class EntityNaming(Metric):
    """Measure the share of entities whose name identifies a thing on its own.

    An entity fails when it has no name, a non-blank string under one of the layout's
    name keys: it cannot be shown to people, resolved as a duplicate, or matched against
    ground truth facts and test questions. It also fails when its name is an English
    pronoun, such as "he" or "they", which refers to an entity only within its sentence:
    the extractor did not resolve the reference to the entity it stands for. And it fails
    when its name has no letter, such as "1923" or "50%": that is a value, which belongs
    in a property of the entity it describes. A name in capitals of two or more letters,
    such as "US" or "IT", is an acronym rather than a pronoun, but a work titled like a
    pronoun, such as the novel "It", fails.
    """

    name: ClassVar[str] = "entity_naming"
    category: ClassVar[MetricCategory] = MetricCategory.INTRINSIC
    unit: ClassVar[str] = "entities"

    def evaluate(self, graph: GraphView) -> MetricResult:
        """Report every entity without a name, or named by a pronoun or a value.

        Args:
            graph: The graph to evaluate.

        Returns:
            The metric's result.
        """
        keys = ", ".join(graph.layout.name_keys)
        findings = [
            Finding(
                message=f"{graph.describe(entity)} {problem}",
                subject=graph.describe(entity),
                node_ids=(entity.id,),
            )
            for entity in graph.entities
            if (problem := _naming_problem(graph.name(entity), keys)) is not None
        ]
        return self.result(len(graph.entities), len(findings), findings)


def _naming_problem(name: str | None, keys: str) -> str | None:
    """Return why a name does not identify a thing on its own, or ``None`` when it does."""
    if name is None:
        return f"has no non-blank string in {keys}"
    letters = sum(char.isalpha() for char in name)
    if not letters:
        return "is named by a value without a letter"
    if normalize_name(name) in _PRONOUNS and not (letters > 1 and name.isupper()):
        return "is named by a pronoun"
    return None
