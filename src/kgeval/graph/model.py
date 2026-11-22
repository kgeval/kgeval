"""Define the immutable property graph snapshot that every metric evaluates."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, time
from typing import Annotated, Literal, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "Duration",
    "EmbeddingDimensions",
    "Graph",
    "Node",
    "Point",
    "PropertyTypeName",
    "PropertyValue",
    "Relationship",
    "Scalar",
    "canonical_json",
    "content_digest",
    "value_type",
]


class Duration(BaseModel):
    """Represent a Neo4j duration, which keeps calendar months apart from exact time."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    months: int = 0
    days: int = 0
    seconds: int = 0
    nanoseconds: int = 0


class Point(BaseModel):
    """Represent a Neo4j spatial point."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    srid: int
    coordinates: tuple[float, ...]


Scalar: TypeAlias = str | bool | int | float | datetime | date | time | Duration | Point
"""A single property value of any type Neo4j stores, except vectors and byte arrays."""

PropertyValue: TypeAlias = Scalar | list[Scalar]
"""A property value: a scalar or a list of scalars."""

PropertyTypeName: TypeAlias = Literal[
    "BOOLEAN",
    "DATE",
    "DURATION",
    "FLOAT",
    "INTEGER",
    "LIST",
    "LOCAL_DATETIME",
    "LOCAL_TIME",
    "POINT",
    "STRING",
    "ZONED_DATETIME",
    "ZONED_TIME",
]
"""The Neo4j property type names, spelled as in neo4j-graphrag schemas."""


def value_type(value: PropertyValue) -> PropertyTypeName:
    """Return the Neo4j type name of a property value.

    Args:
        value: The value to classify.

    Returns:
        The Neo4j type name; ``"LIST"`` for every list.
    """
    # Check subclasses first: bool is an int and datetime is a date.
    if isinstance(value, bool):
        return "BOOLEAN"
    if isinstance(value, int):
        return "INTEGER"
    if isinstance(value, float):
        return "FLOAT"
    if isinstance(value, str):
        return "STRING"
    if isinstance(value, datetime):
        return "LOCAL_DATETIME" if value.tzinfo is None else "ZONED_DATETIME"
    if isinstance(value, date):
        return "DATE"
    if isinstance(value, time):
        return "LOCAL_TIME" if value.tzinfo is None else "ZONED_TIME"
    if isinstance(value, Duration):
        return "DURATION"
    if isinstance(value, Point):
        return "POINT"
    return "LIST"


def canonical_json(value: object) -> str:
    """Encode a value as compact, key-sorted JSON that keeps non-JSON types distinct.

    Values JSON cannot represent are encoded as ``{type name: repr}`` objects. Property
    values are never maps, so these objects cannot collide with real values.

    Args:
        value: Any combination of JSON types and property values.

    Returns:
        The canonical JSON text.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=lambda other: {type(other).__name__: repr(other)},
    )


def content_digest(value: object) -> str:
    """Return the SHA-256 hex digest of a value's canonical JSON.

    Args:
        value: Any combination of JSON types and property values.

    Returns:
        The 64-character hex digest; equal values always share it.
    """
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


EmbeddingDimensions: TypeAlias = dict[str, Annotated[int, Field(ge=1)]]
"""The length of each embedding, a non-empty list of floats, by the property that holds it."""


class Node(BaseModel):
    """Represent a node by its element ID, labels, and properties.

    Attributes:
        id: The node's element ID.
        labels: Its labels.
        properties: Its properties, except embeddings.
        embedding_dimensions: The dimension of each of its embeddings, whose values are
            not kept.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    labels: tuple[str, ...] = ()
    properties: dict[str, PropertyValue] = Field(default_factory=dict)
    embedding_dimensions: EmbeddingDimensions = Field(default_factory=dict)

    @field_validator("labels")
    @classmethod
    def _sort_labels(cls, labels: tuple[str, ...]) -> tuple[str, ...]:
        # Sort labels so equal nodes compare and fingerprint identically.
        return tuple(sorted(set(labels)))


class Relationship(BaseModel):
    """Represent a relationship, which points from its source node to its target node.

    Attributes:
        id: The relationship's element ID.
        type: Its relationship type.
        source_id: The element ID of its source node.
        target_id: The element ID of its target node.
        properties: Its properties, except embeddings.
        embedding_dimensions: The dimension of each of its embeddings, whose values are
            not kept.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    type: str
    source_id: str
    target_id: str
    properties: dict[str, PropertyValue] = Field(default_factory=dict)
    embedding_dimensions: EmbeddingDimensions = Field(default_factory=dict)

    @property
    def node_ids(self) -> tuple[str, ...]:
        """Return the source and target node IDs, once each, so a self-loop yields one ID."""
        return tuple(dict.fromkeys((self.source_id, self.target_id)))


_ElementT = TypeVar("_ElementT", Node, Relationship)


class Graph(BaseModel):
    """Hold an immutable snapshot of a property graph.

    Nodes and relationships are kept sorted by ID, so every computation over a graph
    is deterministic whatever order its source returned them in. Construction fails
    on duplicate IDs and on relationships whose source or target node is missing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    nodes: tuple[Node, ...] = ()
    relationships: tuple[Relationship, ...] = ()

    @field_validator("nodes", "relationships")
    @classmethod
    def _sort_by_id(cls, elements: tuple[_ElementT, ...]) -> tuple[_ElementT, ...]:
        return tuple(sorted(elements, key=lambda element: element.id))

    @model_validator(mode="after")
    def _check_integrity(self) -> Graph:
        node_ids = {node.id for node in self.nodes}
        if len(node_ids) != len(self.nodes):
            raise ValueError("node IDs must be unique")
        relationship_ids = {relationship.id for relationship in self.relationships}
        if len(relationship_ids) != len(self.relationships):
            raise ValueError("relationship IDs must be unique")
        dangling = [
            relationship.id
            for relationship in self.relationships
            if relationship.source_id not in node_ids or relationship.target_id not in node_ids
        ]
        if dangling:
            raise ValueError(
                f"{len(dangling)} relationships reference missing nodes, such as {dangling[0]!r}"
            )
        return self

    def fingerprint(self) -> str:
        """Return a digest that changes when any element, property or embedding dimension does.

        Returns:
            The graph's ``content_digest``, as ``"sha256:<hex>"``.
        """
        nodes = [
            [node.id, node.labels, node.properties, node.embedding_dimensions]
            for node in self.nodes
        ]
        relationships = [
            [
                relationship.id,
                relationship.type,
                relationship.source_id,
                relationship.target_id,
                relationship.properties,
                relationship.embedding_dimensions,
            ]
            for relationship in self.relationships
        ]
        return "sha256:" + content_digest((nodes, relationships))
