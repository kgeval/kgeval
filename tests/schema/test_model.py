"""Test the graph schema in the format neo4j-graphrag uses."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from kgeval.errors import InputError
from kgeval.schema.model import Constraint, GraphSchema, NodeType, load_schema


def test_accepts_neo4j_graphrag_shorthand() -> None:
    schema = GraphSchema.model_validate(
        {
            "node_types": ["Person", "Company"],
            "relationship_types": ["WORKS_AT"],
            "patterns": [("Person", "WORKS_AT", "Company")],
        }
    )

    assert [node_type.label for node_type in schema.node_types] == ["Person", "Company"]
    assert schema.patterns[0].target == "Company"


def test_accepts_neo4j_graphrag_model_dumps() -> None:
    dump: dict[str, Any] = {
        "node_types": [
            {
                "label": "Person",
                "description": "",
                "properties": [
                    {"name": "name", "type": "STRING", "description": "", "required": False}
                ],
                "additional_properties": False,
            }
        ],
        "relationship_types": [
            {"label": "KNOWS", "description": "", "properties": [], "additional_properties": True}
        ],
        "patterns": [{"source": "Person", "relationship": "KNOWS", "target": "Person"}],
        "constraints": [
            {
                "type": "EXISTENCE",
                "property_name": "",
                "property_names": ["name"],
                "node_type": "Person",
                "relationship_type": None,
            }
        ],
        "additional_node_types": False,
        "additional_relationship_types": False,
        "additional_patterns": False,
    }

    schema = GraphSchema.model_validate(dump)

    assert schema.constraints[0].property_names == ("name",)


def test_additional_switches_default_as_in_neo4j_graphrag() -> None:
    open_schema = GraphSchema()
    closed_schema = GraphSchema.model_validate({"node_types": ["Person"]})

    assert (open_schema.additional_node_types, open_schema.additional_patterns) == (True, True)
    assert (closed_schema.additional_node_types, closed_schema.additional_patterns) == (False, True)
    assert NodeType(label="Person").additional_properties is True
    assert (
        NodeType.model_validate(
            {"label": "Person", "properties": [{"name": "name", "type": "STRING"}]}
        ).additional_properties
        is False
    )


def test_deprecated_single_property_name_is_migrated() -> None:
    constraint = Constraint.model_validate(
        {"type": "UNIQUENESS", "property_name": "id", "node_type": "Person"}
    )

    assert constraint.property_names == ("id",)


@pytest.mark.parametrize(
    ("schema", "error"),
    [
        ({"node_types": ["A"], "relationship_types": ["R"], "patterns": [["A", "R", "B"]]}, "'B'"),
        ({"node_types": ["A"], "relationship_types": ["R"], "patterns": [["A", "S", "A"]]}, "'S'"),
        ({"node_types": ["A", "A"]}, "unique"),
        ({"relationship_types": ["R", "R"]}, "relationship types must be unique"),
        (
            {"constraints": [{"type": "UNIQUENESS", "property_name": "x", "node_type": "A"}]},
            "undeclared node type 'A'",
        ),
        (
            {
                "constraints": [
                    {"type": "UNIQUENESS", "property_name": "x", "relationship_type": "R"}
                ]
            },
            "undeclared relationship type 'R'",
        ),
        (
            {"node_types": ["A"], "relationship_types": ["R"], "patterns": [["A", "R", "A"]] * 2},
            "patterns must be unique",
        ),
        ({"node_type": ["A"]}, "node_type"),
        (
            {"node_types": ["A"], "constraints": [{"type": "KEY", "property_names": ["x"]}]},
            "exactly one",
        ),
        ({"relationship_types": [{"label": "R", "max_targets": 0}]}, "max_targets"),
    ],
)
def test_rejects_invalid_schemas(schema: dict[str, Any], error: str) -> None:
    with pytest.raises(ValidationError, match=error):
        GraphSchema.model_validate(schema)


def test_load_schema_reports_invalid_files(tmp_path: Path) -> None:
    path = tmp_path / "schema.yaml"
    path.write_text("node_types: [Person]\npatterns: [[Person, KNOWS, Person]]\n", encoding="utf-8")

    with pytest.raises(InputError, match="undeclared relationship type 'KNOWS'"):
        load_schema(path)


def test_to_yaml_writes_patterns_as_compact_lists() -> None:
    schema = GraphSchema.model_validate(
        {"node_types": ["A"], "relationship_types": ["R"], "patterns": [["A", "R", "A"]]}
    )

    assert "patterns:\n- [A, R, A]\n" in schema.to_yaml()


def test_max_targets_survives_a_yaml_round_trip_and_is_left_out_when_unset(tmp_path: Path) -> None:
    schema = GraphSchema.model_validate(
        {"relationship_types": [{"label": "BORN_IN", "max_targets": 1}, "KNOWS"]}
    )
    path = tmp_path / "schema.yaml"
    path.write_text(schema.to_yaml(), encoding="utf-8")

    assert load_schema(path) == schema
    assert schema.to_yaml().count("max_targets") == 1


def test_load_schema_accepts_a_string_path(tmp_path: Path) -> None:
    path = tmp_path / "schema.yaml"
    path.write_text("node_types: [Person]\n", encoding="utf-8")

    assert load_schema(str(path)).node_types[0].label == "Person"
