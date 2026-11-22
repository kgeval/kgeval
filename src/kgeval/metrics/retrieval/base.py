"""Define the base class of the retrieval metrics."""

from __future__ import annotations

import logging
from typing import ClassVar

from pydantic import Field

from kgeval.errors import InputError
from kgeval.graph.view import GraphView
from kgeval.metrics.base import Metric, MetricCategory
from kgeval.questions import Question, QuestionSet, Support
from kgeval.retrieval import Retrievals

__all__ = ["RetrievalMetric"]

logger = logging.getLogger(__name__)


class RetrievalMetric(Metric):
    """Compare what was retrieved for each test question with what its answer needs.

    The retrieved context holds a supporting fact or entity when it holds a matching
    fact or entity, or a chunk that grounds one, so retrievers that return chunks are
    evaluated as well as those that return graph elements. Each question counts
    equally: the score is the mean, over the questions the metric evaluates, of each
    question's share. ``evaluated`` and ``failed`` count units over all questions.

    The digest of the questions is part of every signature, so results for different
    question sets are never mistaken for each other. The retrievals are what is
    evaluated, like the graph, so they are not part of it.

    Attributes:
        questions: The test questions.
        retrievals: The element IDs retrieved for each question, by question text, in
            rank order.
        k: How many of the top-ranked element IDs of each retrieval to evaluate; all of
            them when omitted.
    """

    category: ClassVar[MetricCategory] = MetricCategory.RETRIEVAL

    questions: QuestionSet
    retrievals: Retrievals
    k: int | None = Field(default=None, ge=1)

    def signature(self) -> str:
        """Return the metric name with a digest of the questions, and ``k`` when set.

        Returns:
            The signature.
        """
        signature = f"{self.name}|questions={self.questions.digest()}"
        return signature if self.k is None else f"{signature}|k={self.k}"

    def retrieved(self, graph: GraphView) -> list[tuple[Question, list[Support], frozenset[str]]]:
        """Pair every question with its support in the graph and what was retrieved for it.

        Args:
            graph: The graph that was retrieved from.

        Returns:
            Each question with its supporting facts and entities, matched as
            ``QuestionSet.support`` matches them, and the top ``k`` retrieved element
            IDs.

        Raises:
            InputError: If a question has no retrieval, or no retrieved element ID is in
                the graph, as when the retrievals come from another database.
        """
        texts = [question.text for question in self.questions.questions]
        if missing := [text for text in texts if text not in self.retrievals]:
            raise InputError(
                f"no retrieval for {len(missing)} of {len(texts)} questions, such as {missing[0]!r}"
            )
        top = {text: frozenset(self.retrievals[text][: self.k]) for text in texts}
        retrieved = {element_id for ids in top.values() for element_id in ids}
        known = {node.id for node in graph.graph.nodes} | {
            relationship.id for relationship in graph.graph.relationships
        }
        if unknown := retrieved - known:
            if unknown == retrieved:
                raise InputError(
                    f"no retrieved element ID is in the graph, such as {min(unknown)!r}; "
                    "retrievals must hold element IDs from the database of the evaluated graph"
                )
            logger.warning(
                "Ignored %d retrieved element IDs that are not in the graph", len(unknown)
            )
        return [
            (question, support, top[question.text])
            for question, support in zip(
                self.questions.questions, self.questions.support(graph), strict=True
            )
        ]
