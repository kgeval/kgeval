"""Test the inference of the schema a graph follows today, as a draft to review."""

from pathlib import Path

from kgeval.metrics.base import Inputs
from kgeval.metrics.schema.conformance import EntityConformance, FactConformance
from kgeval.schema.inference import infer_schema
from kgeval.schema.model import load_schema
from tests.factories import example_view, node, view


def test_infer_schema_orders_types_and_patterns_by_use() -> None:
    schema = infer_schema(example_view())

    assert [node_type.label for node_type in schema.node_types] == [
        "Person",
        "Organization",
        "Landmark",
        "Location",
        "PERSON",
    ]
    person = schema.node_types[0]
    assert [(property_type.name, property_type.type) for property_type in person.properties] == [
        ("born", "DATE"),
        ("id", "STRING"),
    ]
    assert schema.model_dump(mode="json")["patterns"] == [
        ["Person", "FOUNDED", "Organization"],
        ["Person", "BORN_IN", "Location"],
        ["Organization", "FOUNDED", "Location"],
        ["PERSON", "Founded", "Organization"],
        ["Person", "ATTENDED", "Organization"],
        ["Person", "KNOWS", "Person"],
        ["Person", "MARRIED", "Person"],
        ["Person", "STUDIED_AT", "Person"],
    ]


def test_infer_schema_requires_the_properties_every_element_of_a_type_has() -> None:
    schema = infer_schema(example_view())

    assert [
        (constraint.node_type, constraint.property_names) for constraint in schema.constraints
    ] == [
        # Leave born out, since Lee and Fitzwilliam College lack it.
        ("Person", ("id",)),
        ("Landmark", ("id",)),
        ("Location", ("id",)),
        ("PERSON", ("id",)),
    ]


def test_example_conforms_to_its_inferred_schema_except_minority_types(tmp_path: Path) -> None:
    path = tmp_path / "schema.yaml"
    path.write_text(infer_schema(example_view()).to_yaml(), encoding="utf-8")

    schema = load_schema(path)

    entities = EntityConformance().evaluate(example_view(), Inputs(schema=schema))
    assert [finding.message for finding in entities.findings] == [
        '"Kwa Geok Choo" (Person): property born is STRING, expected DATE'
    ]
    assert FactConformance().evaluate(example_view(), Inputs(schema=schema)).failed == 0


def test_inferred_schema_leaves_out_entities_without_a_node_type() -> None:
    assert infer_schema(view(node("a", name="Ada"))).node_types == ()
