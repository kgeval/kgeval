"""Test the evaluation that runs metrics over a graph and reports their results."""

import pytest

from kgeval.errors import InputError
from kgeval.evaluation import assert_passes, evaluate
from kgeval.graph.layout import Layout
from kgeval.ground_truth import load_ground_truth
from kgeval.metrics import (
    BUILTIN_METRICS,
    INTRINSIC_METRICS,
    MODEL_METRICS,
    EntityConnectivity,
    EntityNaming,
    EntityUniqueness,
    FactAccuracy,
    RetrievalRecall,
)
from kgeval.questions import load_questions
from kgeval.schema import load_schema
from tests.doubles import FailingModel, ScriptedModel
from tests.factories import EXAMPLE_DIR, EXAMPLE_RETRIEVALS, example_graph, graph


def test_every_metric_finds_the_planted_defects_of_the_example() -> None:
    report = evaluate(
        example_graph(),
        schema=load_schema(EXAMPLE_DIR / "schema.yaml"),
        ground_truth=load_ground_truth(EXAMPLE_DIR / "ground_truth.yaml"),
        questions=load_questions(EXAMPLE_DIR / "questions.yaml"),
        retrievals=EXAMPLE_RETRIEVALS,
        model=ScriptedModel("-[:KNOWS]->"),
    )

    # Expect extraction_completeness to skip, since the scripted model finds no stated fact.
    failures = {result.metric: (result.failed, result.evaluated) for result in report.results}
    assert failures == {
        "entity_uniqueness": (1, 10),
        "entity_connectivity": (1, 11),
        "entity_naming": (2, 11),
        "entity_provenance": (1, 11),
        "fact_uniqueness": (1, 11),
        "fact_provenance": (1, 11),
        "vocabulary_uniqueness": (2, 14),
        "property_type_consistency": (1, 13),
        "embedding_consistency": (1, 5),
        "extraction_coverage": (1, 5),
        "entity_conformance": (3, 11),
        "fact_conformance": (5, 11),
        "schema_completeness": (3, 13),
        "property_completeness": (3, 15),
        "fact_precision": (2, 5),
        "fact_recall": (1, 4),
        "fact_f1": (3, 9),
        "entity_precision": (1, 5),
        "entity_recall": (1, 5),
        "entity_f1": (2, 10),
        "competency": (1, 5),
        "entity_resolution": (0, 10),
        "entity_disambiguation": (0, 3),
        "node_type_accuracy": (0, 9),
        "fact_accuracy": (1, 9),
        "vocabulary_resolution": (0, 12),
        "extraction_completeness": (0, 0),
        "retrieval_recall": (2, 6),
        "retrieval_precision": (1, 11),
    }
    assert list(failures) == [metric_type.name for metric_type in BUILTIN_METRICS]
    assert report.layout.base_labels == {"__Entity__"}
    assert report.passed is True


def test_without_metrics_every_metric_whose_inputs_are_given_runs() -> None:
    report = evaluate(example_graph(), model=ScriptedModel())

    assert [result.metric for result in report.results] == [
        metric_type.name for metric_type in (*INTRINSIC_METRICS, *MODEL_METRICS)
    ]


def test_given_metrics_need_their_inputs() -> None:
    with pytest.raises(
        InputError,
        match="fact_accuracy needs model; retrieval_recall needs questions and retrievals",
    ):
        evaluate(example_graph(), [FactAccuracy(), RetrievalRecall()])


def test_reports_are_identical_for_identical_graphs() -> None:
    example = example_graph()
    shuffled = graph(*reversed(example.relationships), *reversed(example.nodes))

    assert evaluate(shuffled).model_dump_json() == evaluate(example).model_dump_json()


def test_explicit_layout_replaces_detection() -> None:
    layout = Layout(lexical_labels=frozenset({"Chunk"}), base_labels=frozenset({"__Entity__"}))

    report = evaluate(example_graph(), layout=layout)

    assert (report.summary.lexical_nodes, report.summary.other_nodes) == (5, 3)
    assert report.layout == layout


def test_a_failing_metric_is_reported_as_skipped_and_the_others_still_run() -> None:
    report = evaluate(
        example_graph(), [FactAccuracy(threshold=0.9), EntityNaming()], model=FailingModel()
    )

    reason = "failed: model unavailable: the server is down"
    assert report["fact_accuracy"].skip_reason == reason
    assert report["entity_naming"].failed == 2
    assert report.passed is False
    assert f"**FAIL**, skipped: {reason}" in report.to_markdown()


def test_each_metric_runs_once_per_evaluation() -> None:
    with pytest.raises(InputError, match="repeated: entity_uniqueness"):
        evaluate(example_graph(), [EntityUniqueness(), EntityConnectivity(), EntityUniqueness()])


def test_assert_passes_needs_a_threshold_on_every_metric() -> None:
    with pytest.raises(InputError, match="no metrics to check"):
        assert_passes(example_graph(), [])
    with pytest.raises(InputError, match=r"missing on: entity_naming$"):
        assert_passes(example_graph(), [EntityUniqueness(threshold=0.5), EntityNaming()])


def test_assert_passes_names_each_missed_threshold_on_the_first_line() -> None:
    metrics = [
        EntityUniqueness(threshold=0.95),
        EntityNaming(threshold=0.5),
        FactAccuracy(threshold=0.9),
    ]

    with pytest.raises(AssertionError) as raised:
        assert_passes(example_graph(), metrics, model=FailingModel())

    headline, _, report = str(raised.value).partition("\n\n")
    assert headline == (
        "entity_uniqueness scored 90.0%, below its threshold of 95.0%; "
        "fact_accuracy was skipped: failed: model unavailable: the server is down"
    )
    assert report.startswith("# kgeval report")
