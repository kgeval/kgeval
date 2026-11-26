"""Implement the graph schema in the format neo4j-graphrag uses."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_serializer, model_validator

from kgeval._yaml import load_yaml
from kgeval.graph.model import PropertyTypeName, content_digest

__all__ = [
    "Constraint",
    "GraphSchema",
    "NodeType",
    "Pattern",
    "PropertyType",
    "RelationshipType",
    "load_schema",
]

_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid")


class PropertyType(BaseModel):
    """Declare a property by name and Neo4j type.

    Attributes:
        name: The property key.
        type: The Neo4j type its values must have.
        description: What the property means.
        required: Whether every element of the type must have it. neo4j-graphrag
            deprecates this in favor of EXISTENCE constraints; both are honored.
    """

    model_config = _MODEL_CONFIG

    name: str
    type: PropertyTypeName
    description: str = ""
    required: bool = False


class _ElementType(BaseModel):
    """Declare a node or relationship type by label, with its properties."""

    model_config = _MODEL_CONFIG

    label: str
    description: str = ""
    properties: tuple[PropertyType, ...] = ()
    additional_properties: bool = Field(default_factory=lambda data: not data.get("properties"))

    @model_validator(mode="before")
    @classmethod
    def _from_label(cls, data: object) -> object:
        # Accept a bare label string, as neo4j-graphrag does.
        return {"label": data} if isinstance(data, str) else data


class NodeType(_ElementType):
    """Declare a node type by label, with its properties.

    A bare string is accepted as a node type with that label.

    Attributes:
        label: The node label.
        description: What the type means.
        properties: The declared properties.
        additional_properties: Whether undeclared properties are allowed; defaults to
            true exactly when no properties are declared.
    """


class RelationshipType(_ElementType):
    """Declare a relationship type by label, with its properties.

    A bare string is accepted as a relationship type with that label.

    Attributes:
        label: The relationship type.
        description: What the type means.
        properties: The declared properties.
        additional_properties: Whether undeclared properties are allowed; defaults to
            true exactly when no properties are declared.
        max_targets: The most distinct entities that one entity may relate to with
            this type, such as 1 for ``BORN_IN``; unlimited when omitted. More targets
            contradict each other. This is a kgeval extension of neo4j-graphrag's format.
    """

    max_targets: int | None = Field(default=None, ge=1)


class Pattern(BaseModel):
    """Allow a relationship type between a source and a target node type.

    Accepts and serializes as a ``[source, relationship, target]`` list, and also
    accepts neo4j-graphrag's mapping form.

    Attributes:
        source: The label of the source node type.
        relationship: The relationship type.
        target: The label of the target node type.
    """

    model_config = _MODEL_CONFIG

    source: str
    relationship: str
    target: str

    @model_validator(mode="before")
    @classmethod
    def _from_list(cls, data: object) -> object:
        if isinstance(data, list | tuple) and len(data) == 3:
            return dict(zip(("source", "relationship", "target"), data, strict=True))
        return data

    @model_serializer
    def _to_list(self) -> list[str]:
        return [self.source, self.relationship, self.target]


class Constraint(BaseModel):
    """Declare a uniqueness, existence, or key constraint on a node or relationship type.

    EXISTENCE requires the properties to be present, UNIQUENESS requires their
    combined values to be unique among elements that have them all, and KEY requires
    both.

    Attributes:
        type: The constraint type: EXISTENCE, UNIQUENESS, or KEY.
        property_names: The constrained properties.
        node_type: The label of the constrained node type, when it constrains entities.
        relationship_type: The constrained relationship type, when it constrains facts.
    """

    model_config = _MODEL_CONFIG

    type: Literal["UNIQUENESS", "EXISTENCE", "KEY"]
    property_names: tuple[str, ...] = Field(min_length=1)
    node_type: str | None = None
    relationship_type: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _migrate_property_name(cls, data: object) -> object:
        # Accept neo4j-graphrag's deprecated property_name, which its dumps still include.
        if isinstance(data, Mapping) and "property_name" in data:
            data = dict(data)
            name = data.pop("property_name")
            if name and not data.get("property_names"):
                data["property_names"] = [name]
        return data

    @model_validator(mode="after")
    def _check_scope(self) -> Constraint:
        if (self.node_type is None) == (self.relationship_type is None):
            raise ValueError("a constraint needs exactly one of node_type or relationship_type")
        return self


class GraphSchema(BaseModel):
    """Declare which node types, relationship types, and patterns a graph may contain.

    Each ``additional_*`` switch defaults to true exactly when the matching list is
    empty, as in neo4j-graphrag: declaring node types closes the set of node types
    unless ``additional_node_types`` says otherwise.

    Attributes:
        node_types: The declared node types.
        relationship_types: The declared relationship types.
        patterns: The allowed (source, relationship, target) combinations.
        constraints: Existence, uniqueness, and key constraints.
        additional_node_types: Whether undeclared node labels are allowed.
        additional_relationship_types: Whether undeclared relationship types are allowed.
        additional_patterns: Whether undeclared patterns are allowed.
    """

    model_config = _MODEL_CONFIG

    node_types: tuple[NodeType, ...] = ()
    relationship_types: tuple[RelationshipType, ...] = ()
    patterns: tuple[Pattern, ...] = ()
    constraints: tuple[Constraint, ...] = ()
    additional_node_types: bool = Field(default_factory=lambda data: not data.get("node_types"))
    additional_relationship_types: bool = Field(
        default_factory=lambda data: not data.get("relationship_types")
    )
    additional_patterns: bool = Field(default_factory=lambda data: not data.get("patterns"))

    @model_validator(mode="after")
    def _check_references(self) -> GraphSchema:
        node_labels = [node_type.label for node_type in self.node_types]
        relationship_labels = [relationship.label for relationship in self.relationship_types]
        if len(set(node_labels)) != len(node_labels):
            raise ValueError("node types must be unique")
        if len(set(relationship_labels)) != len(relationship_labels):
            raise ValueError("relationship types must be unique")
        if len(set(self.patterns)) != len(self.patterns):
            raise ValueError("patterns must be unique")
        for pattern in self.patterns:
            for label in (pattern.source, pattern.target):
                if label not in node_labels:
                    raise ValueError(f"pattern uses undeclared node type {label!r}")
            if pattern.relationship not in relationship_labels:
                raise ValueError(
                    f"pattern uses undeclared relationship type {pattern.relationship!r}"
                )
        for constraint in self.constraints:
            if constraint.node_type is not None and constraint.node_type not in node_labels:
                raise ValueError(f"constraint uses undeclared node type {constraint.node_type!r}")
            if (
                constraint.relationship_type is not None
                and constraint.relationship_type not in relationship_labels
            ):
                raise ValueError(
                    f"constraint uses undeclared relationship type {constraint.relationship_type!r}"
                )
        return self

    def to_yaml(self) -> str:
        """Return the schema as YAML that ``load_schema`` reads back.

        Returns:
            The YAML text, leaving out fields equal to a fixed default.
        """
        return yaml.safe_dump(
            self.model_dump(mode="json", exclude_defaults=True),
            sort_keys=False,
            default_flow_style=None,
            allow_unicode=True,
        )

    def digest(self) -> str:
        """Return a short digest that changes whenever the schema changes.

        Returns:
            The first 12 hex digits of the schema's content digest.
        """
        return content_digest(self.model_dump(mode="json"))[:12]


def load_schema(path: str | Path) -> GraphSchema:
    """Load a graph schema from a YAML or JSON file.

    Args:
        path: The schema file.

    Returns:
        The schema.

    Raises:
        InputError: If the file cannot be read or is not a valid schema.
    """
    return load_yaml(path, TypeAdapter(GraphSchema))
