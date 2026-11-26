"""Implement the comparison of two reports, such as a baseline and a new run, metric by metric."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from kgeval.errors import InputError
from kgeval.metrics.base import Finding, MetricCategory, MetricResult
from kgeval.report.markdown import DEFAULT_MAX_FINDINGS, render_comparison_markdown
from kgeval.report.model import Report

__all__ = ["Comparison", "MetricChange", "Trend", "compare"]


class Trend(StrEnum):
    """Name how a metric's score moved from the base report to the head report.

    Attributes:
        IMPROVED: The head scores higher.
        WORSENED: The head scores lower.
        UNCHANGED: Both score the same.
        INCONCLUSIVE: The scores differ, but the 95% interval of the change includes
            zero, so sampling alone can explain the difference.
    """

    IMPROVED = "improved"
    WORSENED = "worsened"
    UNCHANGED = "unchanged"
    INCONCLUSIVE = "inconclusive"


class MetricChange(BaseModel):
    """Pair one metric's results in two reports, with the findings that differ.

    The two results are comparable when both reports ran the metric with the same
    signature; results under different parameters, such as another ground truth,
    measure different things, so their scores and findings are not compared.

    Attributes:
        metric: The metric's name.
        category: The metric's group in reports.
        base: Its result in the base report, or ``None`` when the base did not run it.
        head: Its result in the head report, or ``None`` when the head did not run it.
        new_findings: The head's findings whose subject the base lacks, all of them when
            the base did not run the metric. Always empty for model metrics and for
            results that are not comparable.
        resolved_findings: The base's findings whose subject the head lacks, when the
            head scored the metric. Always empty for model metrics and for results
            that are not comparable.
        max_drop: The largest score drop, from 0 to 1, that is not a regression.
    """

    model_config = ConfigDict(frozen=True)

    metric: str
    category: MetricCategory
    base: MetricResult | None
    head: MetricResult | None
    new_findings: tuple[Finding, ...] = ()
    resolved_findings: tuple[Finding, ...] = ()
    max_drop: float = Field(default=0, ge=0, le=1)

    @property
    def comparable(self) -> bool:
        """Return whether both reports ran the metric with the same parameters."""
        return (
            self.base is not None
            and self.head is not None
            and self.base.signature == self.head.signature
        )

    @property
    def score_change(self) -> float | None:
        """Return the head's score minus the base's, or ``None`` without two comparable scores."""
        if not self.comparable or self.base is None or self.head is None:
            return None
        if self.base.score is None or self.head.score is None:
            return None
        return self.head.score - self.base.score

    @property
    def change_interval(self) -> tuple[float, float] | None:
        """Return a 95% interval for ``score_change`` when either score was estimated from a sample.

        The interval is Newcombe's hybrid score interval, built from each score's
        interval, with a score that has none counting as exact. It treats the two
        samples as independent, which is conservative when they share units.

        Returns:
            The interval, or ``None`` when there is no score change or neither score
            was estimated from a sample.
        """
        change = self.score_change
        if change is None or self.base is None or self.head is None:
            return None
        base_score, head_score = self.base.score, self.head.score
        if base_score is None or head_score is None:
            return None
        if self.base.confidence_interval is None and self.head.confidence_interval is None:
            return None
        base_low, base_high = self.base.confidence_interval or (base_score, base_score)
        head_low, head_high = self.head.confidence_interval or (head_score, head_score)
        return (
            change - math.hypot(head_score - head_low, base_high - base_score),
            change + math.hypot(head_high - head_score, base_score - base_low),
        )

    @property
    def trend(self) -> Trend | None:
        """Return how the score moved, or ``None`` without a score change."""
        change = self.score_change
        if change is None:
            return None
        if change == 0:
            return Trend.UNCHANGED
        interval = self.change_interval
        if interval is not None and interval[0] <= 0 <= interval[1]:
            return Trend.INCONCLUSIVE
        return Trend.IMPROVED if change > 0 else Trend.WORSENED

    @property
    def regressed(self) -> bool:
        """Return whether the head scores worse than the base, or no longer scores the metric.

        The head scores worse when the trend is worsened, so a sampled score must drop by
        more than sampling explains, and the score dropped by more than ``max_drop``.
        For a metric whose findings are compared, the head must also have a new finding
        or more failed units, since a drop that a smaller population alone explains, as
        when removing duplicate facts that passed, adds no defect. New findings without
        a lower score are no regression either: a rebuild that extracts different
        entities and facts fixes some defects as it adds others.
        """
        if self.base is None or self.base.score is None:
            return False
        if self.head is None or self.head.score is None:
            return True
        change = self.score_change
        if change is None or self.trend is not Trend.WORSENED:
            return False
        # Compare with a tolerance, since score differences carry floating point error.
        if -change <= self.max_drop or math.isclose(-change, self.max_drop):
            return False
        if self.category is MetricCategory.MODEL:
            return True
        return bool(self.new_findings) or self.head.failed > self.base.failed


class Comparison(BaseModel):
    """Hold the comparison of a head report with a base report, metric by metric.

    Attributes:
        base: The report compared against, such as a baseline.
        head: The report checked against it.
        changes: One change per metric that either report ran: the head's metrics in
            its order, then the others of the base.
    """

    model_config = ConfigDict(frozen=True)

    base: Report
    head: Report
    changes: tuple[MetricChange, ...]

    @property
    def regressed(self) -> bool:
        """Return whether any metric regressed: scores worse, or is no longer scored."""
        return any(change.regressed for change in self.changes)

    @property
    def trend_counts(self) -> dict[Trend, int]:
        """Count the metrics by how their score moved, leaving out those without a score change.

        Returns:
            The count of every trend, in the order ``Trend`` lists them.
        """
        trends = [change.trend for change in self.changes]
        counts = Counter(trend for trend in trends if trend is not None)
        return {trend: counts[trend] for trend in Trend}

    def to_markdown(self, max_findings: int = DEFAULT_MAX_FINDINGS) -> str:
        """Render the comparison as Markdown: both graphs, score changes, and changed findings.

        Text taken from the graph is escaped, so it cannot inject Markdown.

        Args:
            max_findings: How many new and resolved findings to list per metric; the
                rest are counted.

        Returns:
            The Markdown text.
        """
        return render_comparison_markdown(self, max_findings)

    def _repr_markdown_(self) -> str:
        # Render comparisons as Markdown in notebooks.
        return self.to_markdown()


def compare(base: Report, head: Report, *, max_drop: float = 0.0) -> Comparison:
    """Compare a head report with a base report, metric by metric.

    Results pair up by metric name. Findings of deterministic metrics pair up by
    subject, which names the unit at fault by content, so the same defect matches in
    any run on the same content, even of a rebuilt graph with new element IDs; repeated
    subjects pair up one by one. Findings of model metrics are not compared, since each
    run asks about its own sample and verdicts vary.

    Args:
        base: The report to compare against, such as a baseline.
        head: The report to check against it.
        max_drop: The largest score drop, from 0 to 1, that is not a regression, which
            absorbs the noise of rebuilding a graph with an LLM.

    Returns:
        The comparison.

    Raises:
        InputError: If ``max_drop`` is not between 0 and 1.
    """
    if not 0 <= max_drop <= 1:
        raise InputError(f"max_drop must be from 0 to 1, got {max_drop}")
    base_results = {result.metric: result for result in base.results}
    head_results = {result.metric: result for result in head.results}
    metrics = dict.fromkeys([*head_results, *base_results])
    return Comparison(
        base=base,
        head=head,
        changes=tuple(
            _change(metric, base_results.get(metric), head_results.get(metric), max_drop)
            for metric in metrics
        ),
    )


def _change(
    metric: str, base: MetricResult | None, head: MetricResult | None, max_drop: float
) -> MetricChange:
    category = next(result.category for result in (head, base) if result is not None)
    change = MetricChange(metric=metric, category=category, base=base, head=head, max_drop=max_drop)
    if category is MetricCategory.MODEL or head is None:
        return change
    if base is None:
        return change.model_copy(update={"new_findings": head.findings})
    if not change.comparable:
        return change
    # Resolve nothing when the head skipped: a skip cannot show that a defect was fixed.
    resolved = _unmatched(base.findings, head.findings) if head.score is not None else ()
    return change.model_copy(
        update={
            "new_findings": _unmatched(head.findings, base.findings),
            "resolved_findings": resolved,
        }
    )


def _unmatched(findings: Sequence[Finding], others: Sequence[Finding]) -> tuple[Finding, ...]:
    """Return the findings that no finding of the same subject among the others pairs up with."""
    remaining = Counter(finding.subject for finding in others)
    unmatched: list[Finding] = []
    for finding in findings:
        if remaining[finding.subject]:
            remaining[finding.subject] -= 1
        else:
            unmatched.append(finding)
    return tuple(unmatched)
