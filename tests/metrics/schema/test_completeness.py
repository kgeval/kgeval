"""Test the schema completeness and property completeness metrics."""

from kgeval.metrics.base import Inputs
from kgeval.metrics.schema.completeness import PropertyCompleteness, SchemaCompleteness
from tests.factories import EXAMPLE_SCHEMA, example_schema, example_view


def test_schema_completeness_reports_declared_types_and_patterns_the_graph_lacks() -> None:
    result = SchemaCompleteness().evaluate(example_view(), Inputs(schema=example_schema()))

    # Count 5 node types, 4 relationship types, and 4 patterns.
    assert (result.unit, result.evaluated, result.failed) == ("declared types and patterns", 13, 3)
    assert [finding.message for finding in result.findings] == [
        "no entity has the declared node type Award",
        "no fact has the declared relationship type WON",
        "no fact realizes the declared pattern (Person)-[:WON]->(Award)",
    ]


def test_schema_completeness_needs_patterns_with_matching_endpoint_types() -> None:
    patterns = [["Person", "BORN_IN", "Organization"], *EXAMPLE_SCHEMA["patterns"][1:]]

    result = SchemaCompleteness().evaluate(
        example_view(), Inputs(schema=example_schema(patterns=patterns))
    )

    assert result.findings[2].message == (
        "no fact realizes the declared pattern (Person)-[:BORN_IN]->(Organization)"
    )


def test_schema_completeness_skips_schemas_that_declare_nothing() -> None:
    schema = example_schema(node_types=[], relationship_types=[], patterns=[], constraints=[])

    result = SchemaCompleteness().evaluate(example_view(), Inputs(schema=schema))

    assert result.skip_reason == "no declared types and patterns to evaluate"


def test_every_declared_property_of_an_entity_is_one_unit() -> None:
    result = PropertyCompleteness().evaluate(example_view(), Inputs(schema=example_schema()))

    # Count the units: id and born on 5 people, and id on the 5 other entities of declared types.
    assert (result.unit, result.evaluated, result.failed) == ("declared properties", 15, 3)
    assert [(finding.message, finding.node_ids) for finding in result.findings] == [
        ('"Fitzwilliam College" (Person): declared property born is missing', ("fitzwilliam",)),
        ('"Lee" (Person): declared property born is missing', ("surname",)),
        ("unnamed (Organization): declared property id is missing", ("unnamed",)),
    ]


def test_every_declared_property_of_a_fact_is_one_unit() -> None:
    married = {"label": "MARRIED", "properties": [{"name": "year", "type": "INTEGER"}]}
    schema = example_schema(relationship_types=["BORN_IN", "FOUNDED", married, "WON"])

    result = PropertyCompleteness().evaluate(example_view(), Inputs(schema=schema))

    assert (result.evaluated, result.failed) == (16, 4)
    assert result.findings[3].message == (
        '"Kwa Geok Choo" (Person) -[:MARRIED]-> "Lee Kuan Yew" (Person): '
        "declared property year is missing"
    )
    assert result.findings[3].relationship_ids == ("married",)
