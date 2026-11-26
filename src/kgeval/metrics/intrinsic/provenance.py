"""Implement the entity provenance, fact provenance, and extraction coverage metrics."""

from __future__ import annotations

from typing import ClassVar

from kgeval.graph.view import GraphView
from kgeval.metrics.base import (
    NO_LEXICAL_NODES,
    Finding,
    Inputs,
    Metric,
    MetricCategory,
    MetricResult,
)

__all__ = ["EntityProvenance", "ExtractionCoverage", "FactProvenance"]


class EntityProvenance(Metric):
    """Measure the share of entities grounded in at least one chunk.

    An entity is grounded in a chunk when a provenance relationship joins them. Every
    entity extracted from text should be; an ungrounded entity cannot be checked against
    the text it came from.
    """

    name: ClassVar[str] = "entity_provenance"
    category: ClassVar[MetricCategory] = MetricCategory.INTRINSIC
    unit: ClassVar[str] = "entities"

    def evaluate(self, graph: GraphView, inputs: Inputs) -> MetricResult:
        """Report every entity without a provenance relationship.

        Args:
            graph: The graph to evaluate.
            inputs: The inputs the metric is evaluated with.

        Returns:
            The metric's result.
        """
        if not graph.lexical_nodes:
            return self.skip(inputs, NO_LEXICAL_NODES)
        findings = [
            Finding(
                message=f"{graph.describe(entity)} is grounded in no chunk",
                subject=graph.describe(entity),
                node_ids=(entity.id,),
            )
            for entity in graph.entities
            if not graph.chunk_ids(entity.id)
        ]
        return self.result(inputs, len(graph.entities), len(findings), findings)


class FactProvenance(Metric):
    """Measure the share of facts whose source and target are grounded in a common chunk.

    A builder creates a fact from one chunk, which then grounds both of its entities and
    is the fact's provenance. A fact whose entities share no chunk cannot be traced to
    any text, because it was added by hand, hallucinated across chunks, or left behind
    by entity resolution.
    """

    name: ClassVar[str] = "fact_provenance"
    category: ClassVar[MetricCategory] = MetricCategory.INTRINSIC
    unit: ClassVar[str] = "facts"

    def evaluate(self, graph: GraphView, inputs: Inputs) -> MetricResult:
        """Report every fact whose source and target share no chunk.

        Args:
            graph: The graph to evaluate.
            inputs: The inputs the metric is evaluated with.

        Returns:
            The metric's result.
        """
        if not graph.lexical_nodes:
            return self.skip(inputs, NO_LEXICAL_NODES)
        findings = [
            Finding(
                message=f"no chunk grounds both entities of {graph.describe(fact)}",
                subject=graph.describe(fact),
                node_ids=fact.node_ids,
                relationship_ids=(fact.id,),
            )
            for fact in graph.facts
            if not graph.fact_chunk_ids(fact)
        ]
        return self.result(inputs, len(graph.facts), len(findings), findings)


class ExtractionCoverage(Metric):
    """Measure the share of chunks from which at least one entity was extracted.

    Chunks are as ``GraphView.chunks`` defines them, which in a lexical graph of
    documents and chunks means every chunk and no document. A chunk that grounds no
    entity usually marks a failed or empty extraction.
    """

    name: ClassVar[str] = "extraction_coverage"
    category: ClassVar[MetricCategory] = MetricCategory.INTRINSIC
    unit: ClassVar[str] = "chunks"

    def evaluate(self, graph: GraphView, inputs: Inputs) -> MetricResult:
        """Report every chunk that grounds no entity.

        Args:
            graph: The graph to evaluate.
            inputs: The inputs the metric is evaluated with.

        Returns:
            The metric's result.
        """
        if not graph.lexical_nodes:
            return self.skip(inputs, NO_LEXICAL_NODES)
        if not graph.chunks:
            return self.skip(inputs, "no lexical node has text or grounds an entity")
        grounding = {
            chunk_id for entity in graph.entities for chunk_id in graph.chunk_ids(entity.id)
        }
        findings = [
            Finding(
                message=f"no entity was extracted from {graph.describe(chunk)}",
                subject=graph.describe(chunk),
                node_ids=(chunk.id,),
            )
            for chunk in graph.chunks
            if chunk.id not in grounding
        ]
        return self.result(inputs, len(graph.chunks), len(findings), findings)
