"""Test the pytest plugin that saves the results of assert_passes as a report."""

import pytest

from kgeval.errors import InputError
from kgeval.evaluation import evaluate
from kgeval.metrics import EntityNaming
from kgeval.pytest_plugin import _merge
from kgeval.report import Report
from tests.factories import example_graph

pytest_plugins = ["pytester"]


def test_the_report_lists_each_metric_once_in_name_order_at_its_strictest_threshold(
    pytester: pytest.Pytester,
) -> None:
    pytester.makepyfile(
        """
        import kgeval
        from tests.factories import example_graph

        def test_uniqueness_leniently():
            kgeval.assert_passes(example_graph(), [kgeval.EntityUniqueness(threshold=0.5)])

        def test_naming():
            kgeval.assert_passes(example_graph(), [kgeval.EntityNaming(threshold=0.8)])

        def test_uniqueness():
            kgeval.assert_passes(example_graph(), [kgeval.EntityUniqueness(threshold=0.95)])
        """
    )

    result = pytester.runpytest("--kgeval-report", "scores.json")

    result.assert_outcomes(passed=2, failed=1)
    result.stdout.fnmatch_lines(["*kgeval report written to *scores.json*"])
    report = Report.model_validate_json((pytester.path / "scores.json").read_text("utf-8"))
    assert [result.metric for result in report.results] == ["entity_naming", "entity_uniqueness"]
    assert report["entity_uniqueness"].threshold == 0.95
    assert not report.passed


def test_pytest_fails_without_a_report_when_tests_evaluate_different_graphs(
    pytester: pytest.Pytester,
) -> None:
    pytester.makepyfile(
        """
        import kgeval
        from tests.factories import example_graph, graph, node

        def test_example():
            kgeval.assert_passes(example_graph(), [kgeval.EntityNaming(threshold=0.5)])

        def test_another_graph():
            ada = graph(node("ada", "Person", name="Ada"))
            kgeval.assert_passes(ada, [kgeval.EntityNaming(threshold=0.5)])
        """
    )

    result = pytester.runpytest("--kgeval-report", "scores.json")

    result.assert_outcomes(passed=2)
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.stdout.fnmatch_lines(["*cannot write the kgeval report*the tests evaluated several*"])
    assert not (pytester.path / "scores.json").exists()


def test_a_report_has_one_result_per_metric() -> None:
    report = evaluate(example_graph(), [EntityNaming()])
    changed = report.results[0].model_copy(update={"signature": "entity_naming|changed"})

    with pytest.raises(InputError, match="checked entity_naming with different parameters"):
        _merge([report, report.model_copy(update={"results": (changed,)})])
