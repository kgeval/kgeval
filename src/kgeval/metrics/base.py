"""Define the base class of all metrics, with their categories, findings, and results."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from enum import StrEnum
from typing import Annotated, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from kgeval.graph.model import Node, Relationship
from kgeval.graph.view import GraphView

__all__ = ["NO_LEXICAL_NODES", "Finding", "Metric", "MetricCategory", "MetricResult"]

NO_LEXICAL_NODES = (
    "the graph has no lexical nodes; set Layout.lexical_labels if it has a lexical graph"
)
"""The reason metrics that read chunks give when they skip a graph without a lexical graph."""


def _pass_share(data: dict[str, Any]) -> float | None:
    """Return the share of evaluated units that passed, or ``None`` without a valid share."""
    evaluated, failed = data.get("evaluated"), data.get("failed")
    if not isinstance(evaluated, int) or not isinstance(failed, int) or evaluated == 0:
        return None
    # Leave invalid counts without a share, so the count check reports them.
    return (evaluated - failed) / evaluated if failed <= evaluated else None


class MetricCategory(StrEnum):
    """Name what a metric needs beyond the graph, and so how it decides.

    The first four categories evaluate the graph; the last evaluates retrieval from it.

    Attributes:
        INTRINSIC: Deterministic; needs only the graph.
        SCHEMA: Deterministic; needs a graph schema.
        REFERENCE: Deterministic; needs a ground truth or test questions.
        JUDGED: Needs an LLM judge, so not fully deterministic.
        RETRIEVAL: Deterministic; needs test questions and what a retriever
            returned for each.
    """

    INTRINSIC = "intrinsic"
    SCHEMA = "schema"
    REFERENCE = "reference"
    JUDGED = "judged"
    RETRIEVAL = "retrieval"


class Finding(BaseModel):
    """Describe one defect a metric found, and the exact nodes and relationships involved.

    Attributes:
        message: What is wrong, in plain language. It names elements by content, never
            by element ID, so the same defect has the same message in every run on the
            same graph content, even after a rebuild assigns new element IDs.
        subject: The unit at fault, named by content, such as an entity's name and node
            types. It names neither element IDs nor what is wrong, so it stays the same
            when a rebuild assigns new IDs, when the defect's details change, and when a
            kgeval release rewords the message; ``compare`` matches findings by it.
            Defaults to the message.
        node_ids: Element IDs of the nodes involved.
        relationship_ids: Element IDs of the relationships involved.
    """

    model_config = ConfigDict(frozen=True)

    message: str
    subject: str = Field(default_factory=lambda data: str(data.get("message", "")))
    node_ids: tuple[str, ...] = ()
    relationship_ids: tuple[str, ...] = ()

    @classmethod
    def about(
        cls,
        elements: Node | Relationship | Iterable[Node | Relationship],
        message: str,
        subject: str | None = None,
    ) -> Finding:
        """Return a finding about one or more nodes and relationships.

        Args:
            elements: The node or relationship at fault, or several of them.
            message: What is wrong with them.
            subject: The unit at fault, named by content; the message when omitted.

        Returns:
            The finding, with each element's ID among its node or relationship IDs, in
            the order given.
        """
        # Check for one element first: models iterate over their fields.
        listed = [elements] if isinstance(elements, Node | Relationship) else list(elements)
        return cls(
            message=message,
            subject=message if subject is None else subject,
            node_ids=tuple(element.id for element in listed if isinstance(element, Node)),
            relationship_ids=tuple(
                element.id for element in listed if isinstance(element, Relationship)
            ),
        )


class MetricResult(BaseModel):
    """Hold the outcome of one metric: its score, the counts behind it, and its findings.

    A metric checks each unit of its population, and by default its score is the share
    of units that passed, ``(evaluated - failed) / evaluated``, so 1.0 is perfect. A
    metric may define its score otherwise, such as the mean over questions of each
    question's share, and its docstring then says so. A metric with nothing to evaluate
    is skipped and has no score.

    Attributes:
        metric: The metric's name.
        category: What the metric needs beyond the graph.
        signature: The metric's name and every parameter that affects its score.
        unit: The plural noun for what the metric counts.
        evaluated: How many units were checked.
        failed: How many of them failed.
        score: The score from 0 to 1, where 1 is perfect; ``None`` when skipped.
            Defaults to the share of evaluated units that passed.
        unjudged: How many sampled units the judge gave no usable verdict for, which
            ``evaluated`` leaves out.
        threshold: The minimum passing score, if any.
        confidence_interval: A 95% interval for the score when it was estimated
            from a sample.
        findings: The defects found, with the element IDs involved.
        skip_reason: Why the metric was skipped, when it was.
    """

    model_config = ConfigDict(frozen=True)

    metric: str
    category: MetricCategory
    signature: str
    unit: str
    evaluated: int = Field(ge=0)
    failed: int = Field(ge=0)
    score: Annotated[float, Field(ge=0, le=1)] | None = Field(default_factory=_pass_share)
    unjudged: int = Field(default=0, ge=0)
    threshold: float | None = None
    confidence_interval: tuple[float, float] | None = None
    findings: tuple[Finding, ...] = ()
    skip_reason: str | None = None

    @model_validator(mode="after")
    def _check_counts(self) -> MetricResult:
        if self.failed > self.evaluated:
            raise ValueError("failed cannot exceed evaluated")
        if (self.skip_reason is None) != (self.evaluated > 0):
            raise ValueError("a result is skipped exactly when nothing was evaluated")
        if (self.score is None) != (self.evaluated == 0):
            raise ValueError("a result has a score exactly when something was evaluated")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool | None:
        """Return whether the score meets the threshold; ``None`` without a threshold.

        A skipped metric has no score, so it does not meet a threshold.
        """
        if self.threshold is None:
            return None
        return self.score is not None and self.score >= self.threshold


class Metric(BaseModel, ABC):
    """Measure one quality of a knowledge graph.

    To add a metric, subclass this, set ``name``, ``category``, and ``unit``, declare
    parameters as fields, and implement ``evaluate``, building its return value with
    ``result`` or ``skip``. Override ``signature`` when parameters change the score.
    Give every finding a ``subject`` that names its unit by content, so ``compare``
    matches it across runs.

    Attributes:
        threshold: The minimum passing score; without one the result is informational.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: ClassVar[str]
    category: ClassVar[MetricCategory]
    unit: ClassVar[str]

    threshold: float | None = Field(default=None, ge=0, le=1)

    @abstractmethod
    def evaluate(self, graph: GraphView) -> MetricResult:
        """Evaluate this metric on a graph.

        Args:
            graph: The graph to evaluate.

        Returns:
            The metric's result.
        """

    def signature(self) -> str:
        """Return this metric's name and every parameter that changes its score.

        Returns:
            The name alone, unless a subclass adds its parameters.
        """
        return self.name

    def result(
        self,
        evaluated: int,
        failed: int,
        findings: Iterable[Finding] = (),
        confidence_interval: tuple[float, float] | None = None,
        *,
        score: float | None = None,
        unjudged: int = 0,
    ) -> MetricResult:
        """Build this metric's result, or a skipped one when nothing was evaluated.

        Args:
            evaluated: How many units were checked.
            failed: How many of them failed.
            findings: The defects found.
            confidence_interval: A 95% interval for a score estimated from a sample.
            score: The score, for a metric that does not score the share of evaluated
                units that passed; that share when omitted.
            unjudged: How many sampled units the judge gave no usable verdict for.

        Returns:
            The result.
        """
        if evaluated == 0:
            return self.skip(f"no {self.unit} to evaluate")
        return MetricResult(
            metric=self.name,
            category=self.category,
            signature=self.signature(),
            unit=self.unit,
            evaluated=evaluated,
            failed=failed,
            score=(evaluated - failed) / evaluated if score is None else score,
            unjudged=unjudged,
            threshold=self.threshold,
            confidence_interval=confidence_interval,
            findings=tuple(findings),
        )

    def skip(self, reason: str) -> MetricResult:
        """Build a result recording that this metric does not apply to the graph.

        Args:
            reason: Why the metric does not apply.

        Returns:
            The skipped result.
        """
        return MetricResult(
            metric=self.name,
            category=self.category,
            signature=self.signature(),
            unit=self.unit,
            evaluated=0,
            failed=0,
            threshold=self.threshold,
            skip_reason=reason,
        )
