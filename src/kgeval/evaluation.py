"""Run metrics over a graph and collect their results into a report."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable, Mapping
from importlib.metadata import version

from kgeval.errors import InputError, KgevalError
from kgeval.graph.layout import Layout
from kgeval.graph.model import Graph
from kgeval.graph.view import GraphView
from kgeval.metrics.base import Metric
from kgeval.metrics.catalog import default_metrics
from kgeval.report.model import REPORT_FORMAT, GraphSummary, Report

__all__ = ["assert_passes", "evaluate"]

logger = logging.getLogger(__name__)

# Collect the reports of assert_passes here while pytest runs with --kgeval-report.
_recorded: list[Report] | None = None


def evaluate(
    graph: Graph,
    metrics: Iterable[Metric] | None = None,
    *,
    layout: Layout | None = None,
    metadata: Mapping[str, str] | None = None,
) -> Report:
    """Evaluate a graph with a set of metrics.

    A metric that raises a ``KgevalError`` is reported as skipped, with the error as
    its reason, and the other metrics still run.

    Args:
        graph: The snapshot to evaluate.
        metrics: The metrics to run; ``default_metrics()`` when omitted.
        layout: How to read the graph; detected from its labels when omitted.
        metadata: Describes the run, such as the model or prompt that built the graph;
            the report records it, and ``compare`` shows where two runs differ.

    Returns:
        A report with one result per metric, in the order given.

    Raises:
        InputError: If two metrics share a name.
    """
    selected = default_metrics() if metrics is None else list(metrics)
    counts = Counter(metric.name for metric in selected)
    repeated = sorted(name for name, count in counts.items() if count > 1)
    if repeated:
        raise InputError(
            f"each metric may run once per evaluation, repeated: {', '.join(repeated)}"
        )
    view = GraphView(graph, layout)
    results = []
    for metric in selected:
        logger.info("Evaluating %s", metric.name)
        try:
            results.append(metric.evaluate(view))
        except KgevalError as error:
            logger.warning("Skipped %s: %s", metric.name, error)
            results.append(metric.skip(f"failed: {error}"))
    return Report(
        report_format=REPORT_FORMAT,
        kgeval_version=version("kgeval"),
        graph_fingerprint=graph.fingerprint(),
        layout=view.layout,
        summary=GraphSummary.of(view),
        results=tuple(results),
        metadata=dict(metadata or {}),
    )


def assert_passes(
    graph: Graph, metrics: Iterable[Metric], *, layout: Layout | None = None
) -> Report:
    """Evaluate a graph and fail unless every metric meets its threshold.

    Args:
        graph: The snapshot to evaluate.
        metrics: The metrics to check, each with a threshold.
        layout: How to read the graph; detected from its labels when omitted.

    Returns:
        The report.

    Raises:
        InputError: If there are no metrics, a metric has no threshold, or two metrics
            share a name.
        AssertionError: If a metric misses its threshold.
    """
    # Hide this frame from pytest tracebacks, so a failure points at the test.
    __tracebackhide__ = True
    selected = list(metrics)
    if not selected:
        raise InputError("no metrics to check")
    unchecked = [metric.name for metric in selected if metric.threshold is None]
    if unchecked:
        raise InputError(
            f"each metric to check needs a threshold, missing on: {', '.join(unchecked)}"
        )
    report = evaluate(graph, selected, layout=layout)
    if _recorded is not None:
        _recorded.append(report)
    misses = [
        f"{result.metric} was skipped: {result.skip_reason}"
        if result.score is None
        else f"{result.metric} scored {result.score:.1%}, below its threshold of "
        f"{result.threshold:.1%}"
        for result in report.results
        if result.passed is False
    ]
    if misses:
        # Name every miss on the first line, since pytest's short summary shows only that line.
        raise AssertionError("; ".join(misses) + "\n\n" + report.to_markdown())
    return report
