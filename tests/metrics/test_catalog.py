"""Test listing the built-in metrics and assembling the suite that inputs enable."""

import pytest

from kgeval.errors import InputError
from kgeval.ground_truth import GroundTruth
from kgeval.metrics.base import MetricCategory
from kgeval.metrics.catalog import (
    BUILTIN_METRICS,
    GROUND_TRUTH_METRICS,
    INTRINSIC_METRICS,
    JUDGED_METRICS,
    REFERENCE_METRICS,
    RETRIEVAL_METRICS,
    SCHEMA_METRICS,
    default_metrics,
)
from kgeval.questions import QuestionSet
from kgeval.schema.model import GraphSchema
from tests.doubles import ScriptedJudge

QUESTIONS = QuestionSet.model_validate([{"text": "Who is Ada?", "supporting_entities": ["Ada"]}])


def test_every_builtin_metric_has_a_unique_name_and_sits_in_its_category() -> None:
    groups = {
        MetricCategory.INTRINSIC: INTRINSIC_METRICS,
        MetricCategory.SCHEMA: SCHEMA_METRICS,
        MetricCategory.REFERENCE: REFERENCE_METRICS,
        MetricCategory.JUDGED: JUDGED_METRICS,
        MetricCategory.RETRIEVAL: RETRIEVAL_METRICS,
    }

    names = [metric_type.name for metric_type in BUILTIN_METRICS]

    assert len(names) == len(set(names)) == 29
    for category, members in groups.items():
        assert all(metric_type.category is category for metric_type in members)


def test_default_metrics_without_inputs_are_the_intrinsic_metrics() -> None:
    assert [type(metric) for metric in default_metrics()] == list(INTRINSIC_METRICS)


def test_each_input_adds_the_metrics_that_need_it() -> None:
    schema = GraphSchema.model_validate({"node_types": ["Person"]})
    ground_truth = GroundTruth(facts=(("Ada", "KNOWS", "Alan"),))

    metrics = default_metrics(
        schema=schema,
        ground_truth=ground_truth,
        questions=QUESTIONS,
        retrievals={"Who is Ada?": ("4:x:1",)},
        judge=ScriptedJudge(),
        sample_size=7,
        seed=3,
    )

    assert [type(metric) for metric in metrics] == list(BUILTIN_METRICS)
    signatures = {metric.name: metric.signature() for metric in metrics}
    assert signatures["fact_accuracy"] == "fact_accuracy|judge=scripted|sample_size=7|seed=3"
    assert signatures["vocabulary_resolution"] == "vocabulary_resolution|judge=scripted"
    assert signatures["fact_recall"] == f"fact_recall|ground_truth={ground_truth.digest()}"


def test_retrievals_need_their_questions() -> None:
    with pytest.raises(InputError, match="retrievals need the test questions"):
        default_metrics(retrievals={"Who is Ada?": ()})


def test_ground_truth_metrics_are_reference_metrics() -> None:
    assert set(GROUND_TRUTH_METRICS) < set(REFERENCE_METRICS)
