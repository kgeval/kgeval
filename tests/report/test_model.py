"""Test the report that records the results of one evaluation and how to reproduce them."""

import pytest
from pydantic import ValidationError

from kgeval.evaluation import evaluate
from kgeval.metrics import EntityUniqueness
from kgeval.report.model import REPORT_FORMAT, Report
from tests.factories import example_graph


def test_report_indexes_results_by_metric_name() -> None:
    report = evaluate(example_graph(), [EntityUniqueness()])

    assert report["entity_uniqueness"].failed == 1
    with pytest.raises(KeyError):
        report["fact_accuracy"]


def test_report_survives_a_json_round_trip() -> None:
    report = evaluate(example_graph(), metadata={"chunk_size": "512"})

    assert Report.model_validate_json(report.model_dump_json()) == report


def test_reports_in_an_older_format_are_rejected_with_the_reason() -> None:
    data = evaluate(example_graph(), [EntityUniqueness()]).model_dump(mode="json")
    del data["report_format"]

    with pytest.raises(ValidationError, match="from an older kgeval"):
        Report.model_validate(data)


def test_reports_in_a_newer_format_are_rejected_with_the_reason() -> None:
    data = evaluate(example_graph(), [EntityUniqueness()]).model_dump(mode="json")
    data["report_format"] = REPORT_FORMAT + 1
    data["layout"]["field_from_the_future"] = True

    with pytest.raises(ValidationError, match="which needs a newer kgeval"):
        Report.model_validate(data)
    data["report_format"] = str(REPORT_FORMAT + 1)
    with pytest.raises(ValidationError, match="valid integer"):
        Report.model_validate(data)
