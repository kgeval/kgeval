"""Test the base class of all metrics, with their inputs, categories, findings, and results."""

from typing import ClassVar

import pytest
from pydantic import ValidationError

from kgeval.errors import InputError
from kgeval.graph.view import GraphView
from kgeval.metrics.base import Finding, Inputs, Metric, MetricCategory, MetricResult
from tests.doubles import ScriptedModel
from tests.factories import node, relationship, view


class Fixed(Metric):
    name: ClassVar[str] = "fixed"
    category: ClassVar[MetricCategory] = MetricCategory.INTRINSIC
    unit: ClassVar[str] = "things"

    evaluated: int
    failed: int

    def evaluate(self, graph: GraphView, inputs: Inputs) -> MetricResult:
        return self.result(inputs, self.evaluated, self.failed)


def test_score_is_the_share_of_evaluated_units_that_passed() -> None:
    result = Fixed(evaluated=4, failed=1).evaluate(view(), Inputs())

    assert result.score == 0.75
    assert result.passed is None
    assert result.signature == "fixed"
    assert result.category is MetricCategory.INTRINSIC


def test_metric_without_units_is_skipped_and_unscored() -> None:
    result = Fixed(evaluated=0, failed=0).evaluate(view(), Inputs())

    assert result.skip_reason == "no things to evaluate"
    assert result.score is None
    assert result.passed is None


def test_skipped_metric_misses_its_threshold() -> None:
    assert Fixed(evaluated=0, failed=0, threshold=0.5).evaluate(view(), Inputs()).passed is False


@pytest.mark.parametrize(("failed", "passed"), [(1, True), (2, False)])
def test_passed_compares_score_with_threshold_inclusively(failed: int, passed: bool) -> None:
    metric = Fixed(evaluated=4, failed=failed, threshold=0.75)

    assert metric.evaluate(view(), Inputs()).passed is passed


def test_result_rejects_more_failures_than_evaluations() -> None:
    with pytest.raises(ValidationError, match="failed cannot exceed evaluated"):
        MetricResult(
            metric="m",
            category=MetricCategory.INTRINSIC,
            signature="m",
            unit="things",
            evaluated=1,
            failed=2,
        )


def test_result_is_skipped_exactly_when_nothing_was_evaluated() -> None:
    with pytest.raises(ValidationError, match="skipped exactly"):
        MetricResult(
            metric="m",
            category=MetricCategory.INTRINSIC,
            signature="m",
            unit="things",
            evaluated=0,
            failed=0,
        )


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_threshold_must_be_a_fraction(threshold: float) -> None:
    with pytest.raises(ValidationError):
        Fixed(evaluated=1, failed=0, threshold=threshold)


def test_a_metric_may_score_otherwise_than_the_share_that_passed() -> None:
    result = Fixed(evaluated=4, failed=1).result(Inputs(), 4, 1, score=0.5)

    assert (result.score, result.failed, result.evaluated) == (0.5, 1, 4)


def test_result_has_a_score_exactly_when_something_was_evaluated() -> None:
    with pytest.raises(ValidationError, match="has a score exactly"):
        MetricResult(
            metric="m",
            category=MetricCategory.INTRINSIC,
            signature="m",
            unit="things",
            evaluated=2,
            failed=0,
            score=None,
        )


def test_findings_are_about_their_message_unless_given_a_subject() -> None:
    assert Finding(message="Ada is unnamed").subject == "Ada is unnamed"
    assert Finding(message="Ada is unnamed", subject="Ada").subject == "Ada"


def test_findings_about_relationships_also_list_their_sources_and_targets_once() -> None:
    works_at = relationship("works_at", "ada", "WORKS_AT", "acme")

    finding = Finding.about([works_at, node("ada")], "Ada works at Acme without a start date")

    assert (finding.node_ids, finding.relationship_ids) == (("ada", "acme"), ("works_at",))


def test_a_metric_needs_its_inputs() -> None:
    assert Inputs(model=ScriptedModel()).missing(("schema", "model")) == ["schema"]
    with pytest.raises(InputError, match="fixed needs schema"):
        Fixed(evaluated=1, failed=0).require(Inputs().schema, "schema")
    with pytest.raises(InputError, match="retrievals need their test questions"):
        Inputs(retrievals={})
