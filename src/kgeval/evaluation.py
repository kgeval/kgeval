"""Implement the evaluation that runs metrics over a graph and reports their results."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable, Mapping
from importlib.metadata import version

from kgeval.errors import InputError, KgevalError
from kgeval.graph.layout import Layout
from kgeval.graph.model import Graph
from kgeval.graph.view import GraphView
from kgeval.ground_truth import GroundTruth
from kgeval.metrics.base import Inputs, Metric
from kgeval.metrics.catalog import BUILTIN_METRICS
from kgeval.model import Model
from kgeval.questions import QuestionSet
from kgeval.report.model import REPORT_FORMAT, GraphSummary, Report
from kgeval.retrieval import Retrievals
from kgeval.schema.model import GraphSchema

__all__ = ["assert_passes", "evaluate"]

logger = logging.getLogger(__name__)

# Collect the reports of assert_passes here while pytest runs with --kgeval-report.
_recorded: list[Report] | None = None


def evaluate(
    graph: Graph,
    metrics: Iterable[Metric] | None = None,
    *,
    schema: GraphSchema | None = None,
    ground_truth: GroundTruth | None = None,
    questions: QuestionSet | None = None,
    retrievals: Retrievals | None = None,
    model: Model | None = None,
    layout: Layout | None = None,
    metadata: Mapping[str, str] | None = None,
) -> Report:
    """Evaluate a graph with the given metrics, or every metric whose inputs are given.

    A metric that raises a ``KgevalError`` while it runs is reported as skipped, with
    the error as its reason, and the other metrics still run.

    Args:
        graph: The snapshot to evaluate.
        metrics: The metrics to run; every built-in metric whose inputs are given when
            omitted.
        schema: A graph schema.
        ground_truth: A ground truth.
        questions: Test questions.
        retrievals: What a retriever returned for each test question.
        model: The model that metrics ask when rules cannot decide.
        layout: How to read the graph; detected from its labels when omitted.
        metadata: Describes the run, such as the chunk size or prompt that built the
            graph; the report records it, and ``compare`` shows where two runs differ.

    Returns:
        A report with one result per metric, in the order given.

    Raises:
        InputError: If two metrics share a name, a given metric needs an input that is
            not given, or retrievals are given without their test questions.
    """
    inputs = Inputs(
        schema=schema,
        ground_truth=ground_truth,
        questions=questions,
        retrievals=retrievals,
        model=model,
    )
    if metrics is None:
        selected = [
            metric_type()
            for metric_type in BUILTIN_METRICS
            if not inputs.missing(metric_type.needs)
        ]
    else:
        selected = list(metrics)
        if missing := [
            f"{metric.name} needs {' and '.join(needs)}"
            for metric in selected
            if (needs := inputs.missing(metric.needs))
        ]:
            raise InputError("; ".join(missing))
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
            results.append(metric.evaluate(view, inputs))
        except KgevalError as error:
            logger.warning("Skipped %s: %s", metric.name, error)
            # Flatten the reason, since assert_passes names every miss on its first line.
            results.append(metric.skip(inputs, "failed: " + " ".join(str(error).split())))
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
    graph: Graph,
    metrics: Iterable[Metric],
    *,
    schema: GraphSchema | None = None,
    ground_truth: GroundTruth | None = None,
    questions: QuestionSet | None = None,
    retrievals: Retrievals | None = None,
    model: Model | None = None,
    layout: Layout | None = None,
) -> Report:
    """Evaluate a graph and fail unless every metric meets its threshold.

    Args:
        graph: The snapshot to evaluate.
        metrics: The metrics to check, each with a threshold.
        schema: A graph schema.
        ground_truth: A ground truth.
        questions: Test questions.
        retrievals: What a retriever returned for each test question.
        model: The model that metrics ask when rules cannot decide.
        layout: How to read the graph; detected from its labels when omitted.

    Returns:
        The report.

    Raises:
        InputError: If there are no metrics, a metric has no threshold or needs an input
            that is not given, or two metrics share a name.
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
    report = evaluate(
        graph,
        selected,
        schema=schema,
        ground_truth=ground_truth,
        questions=questions,
        retrievals=retrievals,
        model=model,
        layout=layout,
    )
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
