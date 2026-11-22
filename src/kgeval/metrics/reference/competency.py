"""Implement the competency metric."""

from __future__ import annotations

from typing import ClassVar

from kgeval.graph.view import GraphView
from kgeval.metrics.base import Finding, Metric, MetricCategory, MetricResult
from kgeval.questions import QuestionSet

__all__ = ["Competency"]


class Competency(Metric):
    """Measure the share of test questions whose supporting facts the graph holds.

    A question passes when every one of its supporting facts and entities matches a fact
    or entity of the graph, as ``QuestionSet`` describes; an answer missing any of them
    cannot be right. The retrieval metrics evaluate only what this metric finds in the
    graph, so a failure here is the graph's, and a failure there the retriever's.

    Attributes:
        questions: The test questions.
    """

    name: ClassVar[str] = "competency"
    category: ClassVar[MetricCategory] = MetricCategory.REFERENCE
    unit: ClassVar[str] = "test questions"

    questions: QuestionSet

    def signature(self) -> str:
        """Return the metric name with a digest of the questions.

        Returns:
            The signature.
        """
        return f"{self.name}|questions={self.questions.digest()}"

    def evaluate(self, graph: GraphView) -> MetricResult:
        """Report every question with a supporting fact or entity that the graph lacks.

        Args:
            graph: The graph to evaluate.

        Returns:
            The metric's result.
        """
        findings = [
            Finding(
                message=f"{question.text!r}: the graph lacks {', '.join(missing)}",
                subject=question.text,
            )
            for question, support in zip(
                self.questions.questions, self.questions.support(graph), strict=True
            )
            if (missing := [item.label for item in support if not item.element_ids])
        ]
        return self.result(len(self.questions.questions), len(findings), findings)
