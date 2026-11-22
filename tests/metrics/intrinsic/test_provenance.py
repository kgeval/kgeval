"""Test the entity provenance, fact provenance, and extraction coverage metrics."""

import pytest

from kgeval.metrics.base import Metric
from kgeval.metrics.intrinsic.provenance import (
    EntityProvenance,
    ExtractionCoverage,
    FactProvenance,
)
from tests.factories import example_view, node, relationship, view


@pytest.mark.parametrize("metric", [EntityProvenance(), FactProvenance(), ExtractionCoverage()])
def test_provenance_metrics_skip_graphs_without_a_lexical_graph(metric: Metric) -> None:
    result = metric.evaluate(
        view(node("a", "Person"), node("b", "Movie"), relationship("r", "a", "ACTED_IN", "b"))
    )

    assert result.skip_reason is not None
    assert "no lexical nodes" in result.skip_reason


def test_entity_provenance_reports_entities_grounded_in_no_chunk() -> None:
    result = EntityProvenance().evaluate(example_view())

    assert (result.evaluated, result.failed) == (11, 1)
    [finding] = result.findings
    assert finding.message == '"Marina Bay Sands" (Landmark) is grounded in no chunk'


def test_fact_provenance_needs_a_chunk_that_grounds_both_entities() -> None:
    result = FactProvenance().evaluate(example_view())

    assert (result.evaluated, result.failed) == (11, 1)
    [finding] = result.findings
    assert finding.relationship_ids == ("duplicate_founded",)
    assert finding.node_ids == ("duplicate", "pap")


def test_extraction_coverage_counts_only_lexical_nodes_with_a_chunk_label() -> None:
    result = ExtractionCoverage().evaluate(example_view())

    assert (result.unit, result.evaluated, result.failed) == ("chunks", 5, 1)
    [finding] = result.findings
    assert finding.message == 'no entity was extracted from "Table of contents" (Chunk)'


def test_extraction_coverage_fails_every_chunk_when_no_entity_is_grounded() -> None:
    result = ExtractionCoverage().evaluate(view(node("c", "Chunk", text="Hi"), node("e", "Person")))

    assert (result.evaluated, result.failed) == (1, 1)
