"""Implement the pytest plugin that saves the results of assert_passes as a report."""

from __future__ import annotations

from collections.abc import Generator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from kgeval import evaluation
from kgeval.errors import InputError
from kgeval.metrics.base import MetricResult
from kgeval.report.model import Report

if TYPE_CHECKING:
    from pluggy import Result

__all__ = ["pytest_addoption", "pytest_configure"]


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add the ``--kgeval-report`` option.

    Args:
        parser: The parser of pytest's command-line options.
    """
    parser.getgroup("kgeval").addoption(
        "--kgeval-report",
        metavar="PATH",
        help="Save the results of every assert_passes call to PATH as a JSON report.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Record the reports of assert_passes when ``--kgeval-report`` is set.

    Args:
        config: The pytest configuration.
    """
    path = config.getoption("kgeval_report")
    if path is not None:
        config.pluginmanager.register(_ReportWriter(Path(path).absolute()), "kgeval-report")


class _ReportWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.reports: list[Report] = []
        self.summary: str | None = None
        self.failed = False
        evaluation._recorded = []

    # Use an old-style wrapper, which every pytest supports, since every project loads it.
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self) -> Generator[None, Result[pytest.TestReport], None]:
        outcome = yield
        recorded = evaluation._recorded or []
        if recorded:
            # Attach them to the test report, which pytest-xdist sends from its workers.
            texts = [report.model_dump_json() for report in recorded]
            outcome.get_result().kgeval_reports = texts  # type: ignore[attr-defined]
            recorded.clear()

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        texts = getattr(report, "kgeval_reports", ())
        self.reports += [Report.model_validate_json(text) for text in texts]

    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        # Write from the main process only, as --junitxml does under pytest-xdist.
        if hasattr(session.config, "workerinput"):
            return
        if not self.reports:
            self.summary = "kgeval report not written: no test called assert_passes"
            return
        try:
            text = _merge(self.reports).model_dump_json(indent=2) + "\n"
            self.path.write_text(text, encoding="utf-8")
        except (InputError, OSError) as error:
            self.summary = f"cannot write the kgeval report {self.path}: {error}"
            self.failed = True
            if session.exitstatus == pytest.ExitCode.OK:
                session.exitstatus = pytest.ExitCode.TESTS_FAILED
        else:
            self.summary = f"kgeval report written to {self.path}"

    def pytest_terminal_summary(self, terminalreporter: pytest.TerminalReporter) -> None:
        if self.summary is not None:
            terminalreporter.write_sep("-", self.summary, red=self.failed)

    def pytest_unconfigure(self) -> None:
        evaluation._recorded = None


def _merge(reports: Sequence[Report]) -> Report:
    first = reports[0]
    if any(
        (report.graph_fingerprint, report.layout) != (first.graph_fingerprint, first.layout)
        for report in reports
    ):
        raise InputError("a report covers one graph, but the tests evaluated several")
    results: dict[str, MetricResult] = {}
    for result in (result for report in reports for result in report.results):
        kept = results.setdefault(result.metric, result)
        if kept.signature != result.signature:
            raise InputError(
                "a report has one result per metric, but the tests checked "
                f"{result.metric} with different parameters"
            )
        # Prefer a failing result, then the strictest threshold, so a failed test fails the report.
        if _strictness(result) > _strictness(kept):
            results[result.metric] = result
    # Sort by metric, since pytest-xdist workers finish tests in any order.
    return first.model_copy(update={"results": tuple(results[name] for name in sorted(results))})


def _strictness(result: MetricResult) -> tuple[bool, float]:
    # Rank failing results first, since a model's verdicts can vary from test to test.
    return result.passed is False, result.threshold or 0
