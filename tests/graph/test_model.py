"""Test the immutable property graph snapshot that every metric evaluates."""

from datetime import UTC, date, datetime, time

import pytest
from pydantic import ValidationError

from kgeval.graph.model import (
    Duration,
    Graph,
    Node,
    Point,
    PropertyTypeName,
    PropertyValue,
    canonical_json,
    value_type,
)
from tests.factories import graph, node, relationship


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, "BOOLEAN"),
        (1, "INTEGER"),
        (1.0, "FLOAT"),
        ("1", "STRING"),
        (date(2020, 1, 2), "DATE"),
        (datetime(2020, 1, 2, 3, 4), "LOCAL_DATETIME"),
        (datetime(2020, 1, 2, 3, 4, tzinfo=UTC), "ZONED_DATETIME"),
        (time(3, 4), "LOCAL_TIME"),
        (time(3, 4, tzinfo=UTC), "ZONED_TIME"),
        (Duration(months=1), "DURATION"),
        (Point(srid=4326, coordinates=(1.0, 2.0)), "POINT"),
        (["a", "b"], "LIST"),
    ],
)
def test_value_type_names_every_neo4j_property_type(
    value: PropertyValue, expected: PropertyTypeName
) -> None:
    assert value_type(value) == expected


def test_property_values_keep_their_exact_python_types() -> None:
    moment = datetime(2020, 1, 2, 3, 4)
    properties = node("a", flag=True, count=1, ratio=1.0, moment=moment, day=date(2020, 1, 2))

    assert [type(value) for value in properties.properties.values()] == [
        bool,
        int,
        float,
        datetime,
        date,
    ]


def test_graph_sorts_elements_and_labels() -> None:
    result = graph(
        node("b", "Person", "Author", "Person"),
        node("a"),
        relationship("r2", "a", "KNOWS", "b"),
        relationship("r1", "b", "KNOWS", "a"),
    )

    assert [element.id for element in result.nodes] == ["a", "b"]
    assert [element.id for element in result.relationships] == ["r1", "r2"]
    assert result.nodes[1].labels == ("Author", "Person")


def test_graph_rejects_duplicate_node_ids() -> None:
    with pytest.raises(ValidationError, match="node IDs must be unique"):
        Graph(nodes=(node("a"), node("a")))


def test_graph_models_reject_unknown_fields() -> None:
    # Catch typos such as label for labels, which would otherwise drop the labels silently.
    with pytest.raises(ValidationError, match="label"):
        Node.model_validate({"id": "a", "label": ["Person"]})


def test_graph_rejects_relationships_to_missing_nodes() -> None:
    with pytest.raises(ValidationError, match="missing nodes"):
        graph(node("a"), relationship("r", "a", "KNOWS", "ghost"))


def test_fingerprint_ignores_input_order() -> None:
    first = graph(node("a", name="Ada"), node("b"), relationship("r", "a", "KNOWS", "b"))
    second = graph(relationship("r", "a", "KNOWS", "b"), node("b"), node("a", name="Ada"))

    assert first.fingerprint() == second.fingerprint()
    assert first.fingerprint().startswith("sha256:")


@pytest.mark.parametrize(
    "changed",
    [
        graph(node("a", name="Ada Lovelace"), node("b"), relationship("r", "a", "KNOWS", "b")),
        graph(node("a", "Person", name="Ada"), node("b"), relationship("r", "a", "KNOWS", "b")),
        graph(node("a", name="Ada"), node("b"), relationship("r", "a", "MET", "b")),
        graph(node("a", name="Ada"), node("b"), relationship("r", "b", "KNOWS", "a")),
    ],
)
def test_fingerprint_changes_with_any_content(changed: Graph) -> None:
    original = graph(node("a", name="Ada"), node("b"), relationship("r", "a", "KNOWS", "b"))

    assert changed.fingerprint() != original.fingerprint()


def test_fingerprint_distinguishes_values_that_print_alike() -> None:
    values: list[PropertyValue] = [date(2020, 1, 2), "2020-01-02", 1, 1.0, True, "1"]

    fingerprints = {graph(node("a", value=value)).fingerprint() for value in values}

    assert len(fingerprints) == 6


def test_canonical_json_sorts_keys_and_tags_non_json_types() -> None:
    assert canonical_json({"b": 1, "a": date(2020, 1, 2)}) == '{"a":{"date":"2020-01-02"},"b":1}'


def test_graphs_keep_every_property_type_and_their_fingerprint_through_json() -> None:
    values: list[PropertyValue] = [
        True,
        1,
        1.0,
        "2020-01-02",
        date(2020, 1, 2),
        datetime(2020, 1, 2, 3, 4),
        datetime(2020, 1, 2, 3, 4, 5, 6, tzinfo=UTC),
        time(3, 4),
        time(3, 4, tzinfo=UTC),
        Duration(months=1),
        Point(srid=4326, coordinates=(1.0, 2.0)),
        [date(2020, 1, 2), date(2021, 1, 2)],
    ]
    properties = {f"p{index}": value for index, value in enumerate(values)}
    original = graph(Node(id="a", properties=properties))

    loaded = Graph.model_validate_json(original.model_dump_json())

    assert loaded == original
    assert loaded.fingerprint() == original.fingerprint()
