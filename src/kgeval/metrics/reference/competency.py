"""Implement the competency metric."""

from __future__ import annotations

from typing import ClassVar

from kgeval.graph.view import GraphView
from kgeval.metrics.base import Finding, InputName, Inputs, Metric, MetricCategory, MetricResult

__all__ = ["Competency"]


class Competency(Metric):
    """Measure the share of test questions whose supporting facts the graph holds.

    A question passes when every one of its supporting facts and entities matches a fact
    or entity of the graph, as ``QuestionSet`` describes; an answer missing any of them
    cannot be right. The retrieval metrics evaluate only what this metric finds in the
    graph, so a failure here is the graph's, and a failure there the retriever's.
    """

    name: ClassVar[str] = "competency"
    category: ClassVar[MetricCategory] = MetricCategory.REFERENCE
    unit: ClassVar[str] = "test questions"
    needs: ClassVar[tuple[InputName, ...]] = ("questions",)

    def signature(self, inputs: Inputs) -> str:
        """Return the metric name with a digest of the questions.

        Args:
            inputs: The inputs the metric is evaluated with.

        Returns:
            The signature.
        """
        return f"{self.name}|questions={self.require(inputs.questions, 'questions').digest()}"

    def evaluate(self, graph: GraphView, inputs: Inputs) -> MetricResult:
        """Report every question with a supporting fact or entity that the graph lacks.

        Args:
            graph: The graph to evaluate.
            inputs: The inputs the metric is evaluated with.

        Returns:
            The metric's result.

        Raises:
            InputError: If no test questions were given.
        """
        questions = self.require(inputs.questions, "questions")
        findings = [
            Finding(
                message=f"{question.text!r}: the graph lacks {', '.join(missing)}",
                subject=question.text,
            )
            for question, support in zip(questions.questions, questions.support(graph), strict=True)
            if (missing := [item.label for item in support if not item.element_ids])
        ]
        return self.result(inputs, len(questions.questions), len(findings), findings)
