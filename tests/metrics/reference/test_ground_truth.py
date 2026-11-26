"""Test the fact and entity precision, recall, and F1 metrics."""

import pytest

from kgeval.ground_truth import GroundTruth, load_ground_truth
from kgeval.metrics.base import Inputs, MetricResult
from kgeval.metrics.reference.ground_truth import (
    EntityF1,
    EntityPrecision,
    EntityRecall,
    FactF1,
    FactPrecision,
    FactRecall,
    GroundTruthMetric,
)
from tests.factories import EXAMPLE_DIR, example_view, node, relationship, view

GROUND_TRUTH = load_ground_truth(EXAMPLE_DIR / "ground_truth.yaml")


def measure(
    metric_type: type[GroundTruthMetric], ground_truth: GroundTruth = GROUND_TRUTH
) -> MetricResult:
    return metric_type().evaluate(example_view(), Inputs(ground_truth=ground_truth))


def changed(**fields: object) -> GroundTruth:
    """Return the example ground truth with the given fields replaced, validated again."""
    return GroundTruth.model_validate(GROUND_TRUTH.model_dump() | fields)


@pytest.mark.parametrize(
    ("metric_type", "evaluated", "failed"),
    [
        (FactPrecision, 5, 2),
        (FactRecall, 4, 1),
        (FactF1, 9, 3),
        (EntityPrecision, 5, 1),
        (EntityRecall, 5, 1),
        (EntityF1, 10, 2),
    ],
)
def test_scores_count_distinct_graph_elements_and_ground_truth_elements(
    metric_type: type[GroundTruthMetric], evaluated: int, failed: int
) -> None:
    result = measure(metric_type)

    assert (result.evaluated, result.failed) == (evaluated, failed)


def test_f1_is_the_harmonic_mean_of_precision_and_recall() -> None:
    precision, recall, f1 = (
        measure(metric).score for metric in (FactPrecision, FactRecall, FactF1)
    )

    assert precision is not None
    assert recall is not None
    assert f1 == pytest.approx(2 * precision * recall / (precision + recall))


def test_f1_leaves_its_findings_to_precision_and_recall() -> None:
    assert measure(FactF1).findings == measure(EntityF1).findings == ()


def test_precision_reports_facts_the_ground_truth_lacks_and_skips_unnamed_entities() -> None:
    findings = measure(FactPrecision).findings

    assert [finding.message for finding in findings] == [
        '"He" (PERSON) -[:Founded]-> "People\'s Action Party" (Organization) '
        "is not in the ground truth",
        '"Kwa Geok Choo" (Person) -[:KNOWS]-> "Kwa Geok Choo" (Person) is not in the ground truth',
    ]
    assert findings[1].relationship_ids == ("knows",)


def test_recall_reports_ground_truth_facts_and_entities_the_graph_lacks() -> None:
    assert [finding.message for finding in measure(FactRecall).findings] == [
        'the graph lacks the ground truth fact "Lee Kuan Yew" -[:FOUNDED]-> "Lee & Lee"'
    ]
    assert [finding.message for finding in measure(EntityRecall).findings] == [
        'the graph lacks the ground truth entity "Lee & Lee"'
    ]


def test_entity_precision_counts_names_that_normalize_alike_once() -> None:
    result = measure(EntityPrecision)

    # Count 5 distinct names among 6 entities in scope: "lee kuan yew" duplicates "Lee Kuan Yew".
    assert result.evaluated == 5
    [finding] = result.findings
    assert finding.message == '"He" (PERSON) is not in the ground truth'


def test_documents_limit_the_comparison_to_facts_grounded_in_their_chunks() -> None:
    whole_graph = changed(documents=())

    # Count Marina Bay Sands and the entities of early-years.txt only when it covers all.
    assert measure(EntityPrecision, whole_graph).evaluated == 9
    assert measure(EntityPrecision).evaluated == 5


def test_metrics_skip_when_no_chunk_belongs_to_the_documents() -> None:
    ground_truth = changed(documents=[{"fileName": "missing.txt"}])

    assert measure(FactRecall, ground_truth).skip_reason == (
        "no chunk belongs to the ground truth's documents"
    )


def test_aliases_and_symmetric_types_match_variants_of_ground_truth_facts() -> None:
    graph = view(
        node("lky", "Person", name="Harry Lee"),
        node("kwa", "Person", name="Kwa Geok Choo"),
        node("sg", "Location", name="Singapore"),
        relationship("r1", "kwa", "WED", "lky"),
        relationship("r2", "lky", "BornIn", "sg"),
    )
    ground_truth = GroundTruth(
        facts=(
            ("Lee Kuan Yew", "MARRIED", "Kwa Geok Choo"),
            ("Lee Kuan Yew", "BORN_IN", "Singapore"),
        ),
        aliases={"Lee Kuan Yew": ("Harry Lee",)},
        relationship_type_aliases={"MARRIED": ("WED",)},
        symmetric_relationship_types=("MARRIED",),
    )

    precision = FactPrecision().evaluate(graph, Inputs(ground_truth=ground_truth))
    recall = FactRecall().evaluate(graph, Inputs(ground_truth=ground_truth))

    assert (precision.evaluated, precision.failed, recall.evaluated, recall.failed) == (2, 0, 2, 0)


def test_signature_changes_with_the_ground_truth() -> None:
    fewer = changed(facts=GROUND_TRUTH.facts[1:])

    assert FactRecall().signature(Inputs(ground_truth=GROUND_TRUTH)) != FactRecall().signature(
        Inputs(ground_truth=fewer)
    )
