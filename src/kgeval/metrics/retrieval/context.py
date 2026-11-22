"""Implement the retrieval recall and retrieval precision metrics."""

from __future__ import annotations

from statistics import fmean
from typing import ClassVar

from kgeval.graph.view import GraphView
from kgeval.metrics.base import Finding, MetricResult
from kgeval.metrics.retrieval.base import RetrievalMetric

__all__ = ["RetrievalPrecision", "RetrievalRecall"]


class RetrievalRecall(RetrievalMetric):
    """Measure the share of supporting facts and entities in the graph that were retrieved.

    Only what the graph holds is evaluated, so that the retriever is never blamed for
    a gap in the graph: ``competency`` reports those. A supporting fact counts as
    retrieved when one of its matching facts is, or a chunk that grounds one; retrieving
    both of its entities without the relationship between them does not count. A
    supporting entity counts as retrieved when one of its matching entities is, or a
    chunk that grounds one. The score is the mean of each question's recall, over the
    questions whose support the graph holds. Each finding lists what was missed for one
    question.
    """

    name: ClassVar[str] = "retrieval_recall"
    unit: ClassVar[str] = "supporting facts and entities in the graph"

    def evaluate(self, graph: GraphView) -> MetricResult:
        """Report, for every question, the held supporting facts and entities not retrieved.

        Args:
            graph: The graph that was retrieved from.

        Returns:
            The metric's result.

        Raises:
            InputError: If a question has no retrieval, or no retrieved element ID is in
                the graph.
        """
        recalls: list[float] = []
        evaluated = failed = 0
        findings: list[Finding] = []
        for question, support, retrieved in self.retrieved(graph):
            held = [item for item in support if item.element_ids]
            if not held:
                continue
            missed = [item for item in held if not (item.element_ids | item.chunk_ids) & retrieved]
            recalls.append((len(held) - len(missed)) / len(held))
            evaluated += len(held)
            failed += len(missed)
            if missed:
                findings.append(
                    Finding(
                        message=f"{question.text!r}: the retrieval lacks "
                        + ", ".join(item.label for item in missed),
                        subject=question.text,
                        node_ids=tuple(
                            sorted({element_id for item in missed for element_id in item.node_ids})
                        ),
                        relationship_ids=tuple(
                            sorted(
                                {
                                    element_id
                                    for item in missed
                                    for element_id in item.relationship_ids
                                }
                            )
                        ),
                    )
                )
        if not evaluated:
            return self.skip("the graph holds no supporting fact or entity; competency lists them")
        return self.result(evaluated, failed, findings, score=fmean(recalls))


class RetrievalPrecision(RetrievalMetric):
    """Measure the share of retrieved entities, facts, and chunks that a question's answer needs.

    An entity or fact is needed when it matches a supporting fact or entity of the
    question, or is the source or target of a matching fact; a chunk is needed when it
    grounds one of those matching facts or entities. Other retrieved nodes, such as
    documents and community nodes, are not evaluated. Irrelevant context costs tokens
    and distracts the model that answers. The score is the mean of each question's
    precision, over the questions whose retrieval holds an entity, fact or chunk. Each
    finding lists the unneeded elements of one question.
    """

    name: ClassVar[str] = "retrieval_precision"
    unit: ClassVar[str] = "retrieved entities, facts, and chunks"

    def evaluate(self, graph: GraphView) -> MetricResult:
        """Report, for every question, the retrieved elements its answer does not need.

        Args:
            graph: The graph that was retrieved from.

        Returns:
            The metric's result.

        Raises:
            InputError: If a question has no retrieval, or no retrieved element ID is in
                the graph.
        """
        node_ids = {entity.id for entity in graph.entities} | {chunk.id for chunk in graph.chunks}
        fact_ids = {fact.id for fact in graph.facts}
        precisions: list[float] = []
        evaluated = failed = 0
        findings: list[Finding] = []
        for question, support, retrieved in self.retrieved(graph):
            needed = {
                element_id
                for item in support
                for element_id in item.node_ids | item.relationship_ids | item.chunk_ids
            }
            found = retrieved & (node_ids | fact_ids)
            if not found:
                continue
            unneeded = found - needed
            precisions.append((len(found) - len(unneeded)) / len(found))
            evaluated += len(found)
            failed += len(unneeded)
            if unneeded:
                findings.append(
                    Finding(
                        message=f"{question.text!r}: {len(unneeded)} of {len(found)} retrieved "
                        "entities, facts, and chunks support nothing the answer needs",
                        subject=question.text,
                        node_ids=tuple(sorted(unneeded & node_ids)),
                        relationship_ids=tuple(sorted(unneeded & fact_ids)),
                    )
                )
        if not evaluated:
            return self.skip("no retrieval holds an entity, fact or chunk of the graph")
        return self.result(evaluated, failed, findings, score=fmean(precisions))
