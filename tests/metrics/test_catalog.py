"""Test the catalog of built-in metrics by category, and the loading of suites."""

from pathlib import Path
from typing import Any

import pytest

from kgeval.errors import InputError
from kgeval.ground_truth import load_ground_truth
from kgeval.metrics.base import Inputs, Metric, MetricCategory
from kgeval.metrics.catalog import (
    BUILTIN_METRICS,
    GROUND_TRUTH_METRICS,
    INTRINSIC_METRICS,
    MODEL_METRICS,
    REFERENCE_METRICS,
    RETRIEVAL_METRICS,
    SCHEMA_METRICS,
    load_suite,
)
from kgeval.metrics.intrinsic import EntityUniqueness
from kgeval.metrics.model import EntityCriterion, EntityResolution, FactAccuracy
from kgeval.questions import load_questions
from kgeval.schema.model import load_schema
from tests.doubles import ScriptedModel
from tests.factories import EXAMPLE_DIR, EXAMPLE_RETRIEVALS, example_view

EXAMPLE_INPUTS: dict[str, Any] = {
    "schema": load_schema(EXAMPLE_DIR / "schema.yaml"),
    "ground_truth": load_ground_truth(EXAMPLE_DIR / "ground_truth.yaml"),
    "questions": load_questions(EXAMPLE_DIR / "questions.yaml"),
    "retrievals": EXAMPLE_RETRIEVALS,
    "model": ScriptedModel(),
}


def test_every_builtin_metric_has_a_unique_name_and_sits_in_its_category() -> None:
    groups = {
        MetricCategory.INTRINSIC: INTRINSIC_METRICS,
        MetricCategory.SCHEMA: SCHEMA_METRICS,
        MetricCategory.REFERENCE: REFERENCE_METRICS,
        MetricCategory.MODEL: MODEL_METRICS,
        MetricCategory.RETRIEVAL: RETRIEVAL_METRICS,
    }

    names = [metric_type.name for metric_type in BUILTIN_METRICS]

    assert len(names) == len(set(names)) == 29
    for category, members in groups.items():
        assert all(metric_type.category is category for metric_type in members)


def test_ground_truth_metrics_are_reference_metrics() -> None:
    assert set(GROUND_TRUTH_METRICS) < set(REFERENCE_METRICS)


@pytest.mark.parametrize("metric_type", BUILTIN_METRICS, ids=lambda metric_type: metric_type.name)
def test_every_builtin_metric_runs_with_only_the_inputs_it_needs(
    metric_type: type[Metric],
) -> None:
    inputs = Inputs(**{name: EXAMPLE_INPUTS[name] for name in metric_type.needs})

    assert metric_type().evaluate(example_view(), inputs).metric == metric_type.name


def write_suite(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "suite.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_suite_builds_the_listed_metrics_with_their_settings_in_order(
    tmp_path: Path,
) -> None:
    path = write_suite(
        tmp_path,
        "fact_accuracy: {threshold: 0.9, sample_size: 5}\n"
        "entity_uniqueness:\n"
        "entity_resolution: {threshold: 0.8}\n"
        "dated_people: {criterion: It gives a birth date., applies_to: entities}\n",
    )

    assert load_suite(path) == [
        FactAccuracy(threshold=0.9, sample_size=5),
        EntityUniqueness(),
        EntityResolution(threshold=0.8),
        EntityCriterion(name="dated_people", criterion="It gives a birth date."),
    ]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{}\n", "at least 1 item"),
        ("fact_acuracy: {threshold: 0.9}\n", "unknown metric fact_acuracy; did you mean"),
        ("entity_uniqueness: {threshold: 2}\n", "less than or equal to 1"),
        ("entity_uniqueness: {treshold: 0.9}\n", "Extra inputs"),
        ("entity_uniqueness: {threshold: 0.9}\nentity_uniqueness:\n", "repeated keys"),
        ("fact_accuracy: {criterion: It is right., applies_to: facts}\n", "no built-in"),
        ("dated: {criterion: It is dated., applies_to: facts, name: x}\n", "its entry"),
        ("dated: {criterion: It is dated.}\n", "Unable to extract tag"),
    ],
)
def test_load_suite_rejects_invalid_suites(tmp_path: Path, content: str, message: str) -> None:
    with pytest.raises(InputError, match=r"suite\.yaml") as error:
        load_suite(write_suite(tmp_path, content))
    assert message in str(error.value)
