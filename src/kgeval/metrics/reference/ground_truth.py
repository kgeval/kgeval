"""Implement the fact and entity precision, recall, and F1 metrics."""

from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from typing import ClassVar

from kgeval.graph.model import Node, Relationship
from kgeval.graph.view import GraphView
from kgeval.ground_truth import GroundTruth
from kgeval.metrics.base import Finding, Metric, MetricCategory, MetricResult

__all__ = [
    "EntityF1",
    "EntityPrecision",
    "EntityRecall",
    "FactF1",
    "FactPrecision",
    "FactRecall",
    "GroundTruthMetric",
]


class GroundTruthMetric(Metric):
    """Compare the graph with a ground truth of facts, as precision, recall or F1.

    Entity names and relationship types match under the keys that the ground truth's
    ``entity_key`` and ``fact_key`` return: normalized, with aliases mapped to the ground truth
    names they stand for, and in either direction for symmetric relationship types.
    Both sides are compared as sets, so duplicates count once. Facts and entities
    involving an entity without a name cannot match and are not evaluated;
    ``entity_naming`` reports them. When the ground truth names documents, only the
    facts and entities grounded in their chunks are compared. Precision scores the
    graph's distinct facts or entities, failing those the ground truth lacks, so it
    assumes the ground truth lists every fact that its documents, or without documents
    the chunks, state. Recall scores the ground truth facts or entities, failing those the graph
    lacks. F1 scores both, which makes its score ``2 * matched / (distinct + ground_truth)``, the
    harmonic mean of precision and recall; its findings are those of precision and
    recall, so it reports none of its own.

    Attributes:
        ground_truth: The ground truth.
    """

    category: ClassVar[MetricCategory] = MetricCategory.REFERENCE
    scores_graph: ClassVar[bool]
    scores_ground_truth: ClassVar[bool]

    ground_truth: GroundTruth

    def signature(self) -> str:
        """Return the metric name with a digest of the ground truth.

        Returns:
            The signature.
        """
        return f"{self.name}|ground_truth={self.ground_truth.digest()}"

    def evaluate(self, graph: GraphView) -> MetricResult:
        """Match distinct facts or entities with the ground truth; report each unmatched.

        Args:
            graph: The graph to evaluate.

        Returns:
            The metric's result.
        """
        scope: frozenset[str] | None = None
        if self.ground_truth.documents:
            scope = graph.chunks_of(self.ground_truth.documents)
            if not scope:
                return self.skip("no chunk belongs to the ground truth's documents")
        extracted = self._extracted(graph, scope)
        expected = self._expected()
        evaluated = 0
        unmatched: list[Finding] = []
        if self.scores_graph:
            evaluated += len(extracted)
            unmatched += [finding for key, finding in extracted.items() if key not in expected]
        if self.scores_ground_truth:
            evaluated += len(expected)
            unmatched += [finding for key, finding in expected.items() if key not in extracted]
        # Leave F1's findings to precision and recall, which report the same defects.
        findings = () if self.scores_graph and self.scores_ground_truth else unmatched
        return self.result(evaluated, len(unmatched), findings)

    @abstractmethod
    def _extracted(
        self, graph: GraphView, scope: frozenset[str] | None
    ) -> dict[tuple[str, ...], Finding]:
        """Map each distinct key in scope to the finding it gets when unmatched."""

    @abstractmethod
    def _expected(self) -> dict[tuple[str, ...], Finding]:
        """Map each distinct ground_truth key to the finding it gets when unmatched."""


class _FactMetric(GroundTruthMetric):
    def _extracted(
        self, graph: GraphView, scope: frozenset[str] | None
    ) -> dict[tuple[str, ...], Finding]:
        groups: defaultdict[tuple[str, ...], list[Relationship]] = defaultdict(list)
        for fact in graph.facts:
            if scope is not None and not graph.fact_chunk_ids(fact) & scope:
                continue
            key = self.ground_truth.graph_fact_key(graph, fact)
            if all(key):
                groups[key].append(fact)
        # Describe each group by its alphabetically first description, which no element ID changes.
        return {
            key: Finding(
                message=f"{min(map(graph.describe, facts))} is not in the ground truth",
                subject=_fact_subject(key),
                node_ids=tuple(
                    dict.fromkeys(node_id for fact in facts for node_id in fact.node_ids)
                ),
                relationship_ids=tuple(fact.id for fact in facts),
            )
            for key, facts in groups.items()
        }

    def _expected(self) -> dict[tuple[str, ...], Finding]:
        expected: dict[tuple[str, ...], Finding] = {}
        for source, relationship_type, target in self.ground_truth.facts:
            key = self.ground_truth.fact_key(source, relationship_type, target)
            expected.setdefault(
                key,
                Finding(
                    message="the graph lacks the ground truth fact "
                    f'"{source}" -[:{relationship_type}]-> "{target}"',
                    subject=_fact_subject(key),
                ),
            )
        return expected


class _EntityMetric(GroundTruthMetric):
    def _extracted(
        self, graph: GraphView, scope: frozenset[str] | None
    ) -> dict[tuple[str, ...], Finding]:
        groups: defaultdict[tuple[str, ...], list[Node]] = defaultdict(list)
        for entity in graph.entities:
            if scope is not None and not graph.chunk_ids(entity.id) & scope:
                continue
            if key := self.ground_truth.entity_key(graph.name(entity) or ""):
                groups[(key,)].append(entity)
        return {
            key: Finding(
                message=f"{min(map(graph.describe, entities))} is not in the ground truth",
                subject=key[0],
                node_ids=tuple(entity.id for entity in entities),
            )
            for key, entities in groups.items()
        }

    def _expected(self) -> dict[tuple[str, ...], Finding]:
        expected: dict[tuple[str, ...], Finding] = {}
        for name in self.ground_truth.named_entities():
            key = self.ground_truth.entity_key(name)
            expected.setdefault(
                (key,),
                Finding(message=f'the graph lacks the ground truth entity "{name}"', subject=key),
            )
        return expected


def _fact_subject(key: tuple[str, ...]) -> str:
    """Name a fact by its matching key, in Cypher arrow notation."""
    source, relationship_type, target = key
    return f"{source} -[:{relationship_type}]-> {target}"


class FactPrecision(_FactMetric):
    """Measure the share of the graph's distinct facts that the ground truth contains.

    A fact matches a ground truth fact when its source entity's name, relationship type, and
    target entity's name match the ground truth fact's, in that direction unless the
    relationship type is symmetric.
    """

    name: ClassVar[str] = "fact_precision"
    unit: ClassVar[str] = "distinct facts"
    scores_graph: ClassVar[bool] = True
    scores_ground_truth: ClassVar[bool] = False


class FactRecall(_FactMetric):
    """Measure the share of ground truth facts that the graph contains.

    Facts match ground truth facts as in ``fact_precision``.
    """

    name: ClassVar[str] = "fact_recall"
    unit: ClassVar[str] = "ground truth facts"
    scores_graph: ClassVar[bool] = False
    scores_ground_truth: ClassVar[bool] = True


class FactF1(_FactMetric):
    """Measure the harmonic mean of ``fact_precision`` and ``fact_recall``.

    It reports no findings: ``fact_precision`` and ``fact_recall`` list the facts it
    counts as failed.
    """

    name: ClassVar[str] = "fact_f1"
    unit: ClassVar[str] = "distinct and ground truth facts"
    scores_graph: ClassVar[bool] = True
    scores_ground_truth: ClassVar[bool] = True


class EntityPrecision(_EntityMetric):
    """Measure the share of the graph's distinct entities that the ground truth contains.

    The ground truth entities are the source and target entities of the ground truth facts, and
    entities match them by name.
    """

    name: ClassVar[str] = "entity_precision"
    unit: ClassVar[str] = "distinct entities"
    scores_graph: ClassVar[bool] = True
    scores_ground_truth: ClassVar[bool] = False


class EntityRecall(_EntityMetric):
    """Measure the share of ground truth entities that the graph contains.

    Ground truth entities match entities as in ``entity_precision``.
    """

    name: ClassVar[str] = "entity_recall"
    unit: ClassVar[str] = "ground truth entities"
    scores_graph: ClassVar[bool] = False
    scores_ground_truth: ClassVar[bool] = True


class EntityF1(_EntityMetric):
    """Measure the harmonic mean of ``entity_precision`` and ``entity_recall``.

    It reports no findings: ``entity_precision`` and ``entity_recall`` list the entities
    it counts as failed.
    """

    name: ClassVar[str] = "entity_f1"
    unit: ClassVar[str] = "distinct and ground truth entities"
    scores_graph: ClassVar[bool] = True
    scores_ground_truth: ClassVar[bool] = True
