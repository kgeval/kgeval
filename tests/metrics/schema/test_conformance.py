"""Test the entity conformance and fact conformance metrics."""

from typing import Any

import pytest

from kgeval.metrics.base import MetricResult
from kgeval.metrics.schema.base import SchemaMetric
from kgeval.metrics.schema.conformance import EntityConformance, FactConformance
from tests.factories import EXAMPLE_SCHEMA, example_schema, example_view, node, view


def check(metric_type: type[SchemaMetric], **changes: Any) -> MetricResult:
    return metric_type(graph_schema=example_schema(**changes)).evaluate(example_view())


def messages(result: MetricResult) -> list[str]:
    return [finding.message for finding in result.findings]


def node_types_with(node_type: dict[str, Any]) -> list[Any]:
    """Return the example's node types with the one of the same label replaced."""
    return [
        node_type if item["label"] == node_type["label"] else item
        for item in EXAMPLE_SCHEMA["node_types"]
    ]


def test_entities_need_declared_property_types_and_satisfied_constraints() -> None:
    result = check(EntityConformance)

    assert (result.unit, result.evaluated) == ("entities", 11)
    assert messages(result) == [
        '"He" (PERSON): undeclared node type PERSON',
        '"Kwa Geok Choo" (Person): property born is STRING, expected DATE',
        "unnamed (Organization): KEY constraint on Organization needs property id",
    ]


def test_facts_need_a_declared_relationship_type() -> None:
    result = check(FactConformance)

    # Pass the two copies of BORN_IN: its max_targets limits distinct targets, not facts.
    assert (result.unit, result.evaluated) == ("facts", 11)
    assert messages(result) == [
        '"Lee Kuan Yew" (Person) -[:ATTENDED]-> "Raffles" (Organization): '
        "undeclared relationship type ATTENDED",
        '"He" (PERSON) -[:Founded]-> "People\'s Action Party" (Organization): '
        "undeclared relationship type Founded",
        '"Kwa Geok Choo" (Person) -[:KNOWS]-> "Kwa Geok Choo" (Person): '
        "undeclared relationship type KNOWS",
        '"Raffles" (Organization) -[:FOUNDED]-> "Singapore" (Location): '
        "pattern (Organization)-[:FOUNDED]->(Location) is not declared",
        '"Lee" (Person) -[:STUDIED_AT]-> "Fitzwilliam College" (Person): '
        "undeclared relationship type STUDIED_AT",
    ]
    assert result.findings[2].relationship_ids == ("knows",)


def test_undeclared_node_types_fail_unless_additional_node_types_are_allowed() -> None:
    node_types = [item for item in EXAMPLE_SCHEMA["node_types"] if item["label"] != "Landmark"]

    closed = check(EntityConformance, node_types=node_types)
    open_ = check(EntityConformance, node_types=node_types, additional_node_types=True)

    assert '"Marina Bay Sands" (Landmark): undeclared node type Landmark' in messages(closed)
    assert (closed.failed, open_.failed) == (4, 2)


def test_entities_without_a_node_type_fail_closed_node_types() -> None:
    result = EntityConformance(graph_schema=example_schema()).evaluate(
        view(node("a", name="Thing"))
    )

    assert messages(result) == ['"Thing": no node type']


def test_embeddings_count_as_present_list_properties() -> None:
    person = {
        "label": "Person",
        "properties": [
            {"name": "name", "type": "STRING"},
            {"name": "embedding", "type": "LIST", "required": True},
        ],
    }
    constraint = {"type": "EXISTENCE", "property_names": ["embedding"], "node_type": "Person"}
    schema = example_schema(node_types=node_types_with(person), constraints=[constraint])

    result = EntityConformance(graph_schema=schema).evaluate(
        view(node("ada", "Person", name="Ada", embeddings={"embedding": 3}))
    )

    assert (result.evaluated, result.failed) == (1, 0)


def test_required_properties_must_be_present() -> None:
    organization = {
        "label": "Organization",
        "properties": [{"name": "id", "type": "STRING", "required": True}],
    }

    result = check(EntityConformance, node_types=node_types_with(organization), constraints=[])

    assert "unnamed (Organization): required property id is missing" in messages(result)


def test_undeclared_properties_fail_unless_the_type_allows_them() -> None:
    person = {"label": "Person", "properties": [{"name": "id", "type": "STRING"}]}

    closed = check(EntityConformance, node_types=node_types_with(person))
    open_ = check(
        EntityConformance, node_types=node_types_with(person | {"additional_properties": True})
    )

    assert '"Lee Kuan Yew" (Person): undeclared property born' in messages(closed)
    assert (closed.failed, open_.failed) == (5, 2)


def test_facts_need_a_declared_pattern() -> None:
    patterns = [pattern for pattern in EXAMPLE_SCHEMA["patterns"] if pattern[1] != "MARRIED"]

    result = check(FactConformance, patterns=patterns)

    assert (
        '"Kwa Geok Choo" (Person) -[:MARRIED]-> "Lee Kuan Yew" (Person): '
        "pattern (Person)-[:MARRIED]->(Person) is not declared"
    ) in messages(result)


def test_facts_fail_when_their_source_has_more_targets_than_their_type_allows() -> None:
    relationship_types = ["BORN_IN", {"label": "FOUNDED", "max_targets": 1}, "MARRIED", "WON"]

    result = check(FactConformance, relationship_types=relationship_types)

    # Fail both facts from Lee Kuan Yew, but not the one from his duplicate.
    contradicting = [
        finding.relationship_ids
        for finding in result.findings
        if finding.message.endswith("FOUNDED allows 1 target per source, and its source has 2")
    ]
    assert contradicting == [("founded_pap",), ("founded_unnamed",)]


@pytest.mark.parametrize("constraint_type", ["UNIQUENESS", "KEY"])
def test_uniqueness_flags_every_element_that_shares_its_values(constraint_type: str) -> None:
    constraint = {"type": constraint_type, "property_names": ["born"], "node_type": "Person"}

    result = check(EntityConformance, constraints=[constraint])

    unique = (
        f"{constraint_type} constraint on Person needs unique born, shared with 1 other element"
    )
    # Fail the people without born too, but only for KEY, which also requires the property.
    missing = f"{constraint_type} constraint on Person needs property born"
    assert messages(result) == [
        f'"lee kuan yew" (Person): {unique}',
        *([f'"Fitzwilliam College" (Person): {missing}'] if constraint_type == "KEY" else []),
        '"He" (PERSON): undeclared node type PERSON',
        '"Kwa Geok Choo" (Person): property born is STRING, expected DATE',
        f'"Lee Kuan Yew" (Person): {unique}',
        *([f'"Lee" (Person): {missing}'] if constraint_type == "KEY" else []),
    ]


def test_existence_constraints_apply_to_relationship_types() -> None:
    constraint = {"type": "EXISTENCE", "property_names": ["year"], "relationship_type": "MARRIED"}

    facts = check(FactConformance, constraints=[constraint])
    entities = check(EntityConformance, constraints=[constraint])

    assert (
        '"Kwa Geok Choo" (Person) -[:MARRIED]-> "Lee Kuan Yew" (Person): '
        "EXISTENCE constraint on MARRIED needs property year"
    ) in messages(facts)
    assert entities.failed == 2


def test_signature_identifies_the_schema() -> None:
    first = EntityConformance(graph_schema=example_schema())
    second = EntityConformance(graph_schema=example_schema(patterns=[]))

    assert first.signature().startswith("entity_conformance|schema=")
    assert first.signature() != second.signature()
