"""Implement the source that loads knowledge graphs from Neo4j."""

from __future__ import annotations

import logging
from collections.abc import Callable

import neo4j
import neo4j.spatial
import neo4j.time
from neo4j.exceptions import DriverError, Neo4jError

from kgeval.errors import SourceError
from kgeval.graph.model import Duration, Graph, Node, Point, PropertyValue, Relationship, Scalar

__all__ = ["Neo4jSource"]

logger = logging.getLogger(__name__)

_SYNTAX_ERROR = "Neo.ClientError.Statement.SyntaxError"

# Read non-empty float lists, which are embeddings, as their sizes; [] also matches LIST<FLOAT>.
_NODES_QUERY = """
MATCH (node)
RETURN elementId(node) AS id,
       labels(node) AS labels,
       [key IN keys(node) WHERE node[key] = [] OR node[key] IS NOT :: LIST<FLOAT>
        | [key, node[key]]] AS properties,
       [key IN keys(node) WHERE node[key] <> [] AND node[key] IS :: LIST<FLOAT>
        | [key, size(node[key])]] AS embedding_dimensions
"""

_RELATIONSHIPS_QUERY = """
MATCH (source)-[relationship]->(target)
RETURN elementId(relationship) AS id,
       type(relationship) AS type,
       elementId(source) AS source_id,
       elementId(target) AS target_id,
       [key IN keys(relationship)
        WHERE relationship[key] = [] OR relationship[key] IS NOT :: LIST<FLOAT>
        | [key, relationship[key]]] AS properties,
       [key IN keys(relationship)
        WHERE relationship[key] <> [] AND relationship[key] IS :: LIST<FLOAT>
        | [key, size(relationship[key])]] AS embedding_dimensions
"""


class Neo4jSource:
    """Load graph snapshots from a Neo4j database.

    Snapshots keep only the length of each non-empty float-list property, which holds an
    embedding, and leave out vector and byte-array values and temporal values with years
    outside 1 to 9999, which Python cannot represent. Requires Neo4j 5.10 or later,
    the first release whose type predicates accept lists.

    Args:
        driver: An open driver; the caller owns it and closes it.
        database: The database to read; the server's default database when omitted.
    """

    def __init__(self, driver: neo4j.Driver, database: str | None = None) -> None:
        self._driver = driver
        self._database = database

    def load(self) -> Graph:
        """Read every node and relationship into a graph snapshot.

        Nodes and relationships are read in one transaction. Relationships created
        between the two reads, whose nodes the snapshot lacks, are dropped with a
        warning.

        Returns:
            The snapshot.

        Raises:
            SourceError: If the database cannot be read.
        """
        try:
            with self._driver.session(
                database=self._database, default_access_mode=neo4j.READ_ACCESS
            ) as session:
                nodes, relationships = session.execute_read(_read_graph)
        except (Neo4jError, DriverError) as error:
            if isinstance(error, Neo4jError) and error.code == _SYNTAX_ERROR:
                # Name the cause, since the snapshot queries are valid Cypher from Neo4j 5.10 on.
                raise SourceError(
                    "Neo4j rejected the snapshot query; kgeval needs Neo4j 5.10 or later "
                    f"({_describe(error).splitlines()[0]})"
                ) from error
            raise SourceError(f"cannot read the graph from Neo4j: {_describe(error)}") from error
        node_ids = {node.id for node in nodes}
        kept = [
            relationship
            for relationship in relationships
            if relationship.source_id in node_ids and relationship.target_id in node_ids
        ]
        if len(kept) < len(relationships):
            logger.warning(
                "Dropped %d relationships to nodes created while loading",
                len(relationships) - len(kept),
            )
        logger.info("Loaded %d nodes and %d relationships", len(nodes), len(kept))
        return Graph(nodes=tuple(nodes), relationships=tuple(kept))


def _describe(error: Neo4jError | DriverError) -> str:
    # Prefer the server's message, since str() of a server error repeats it among status codes.
    if isinstance(error, Neo4jError) and error.message:
        return error.message
    return str(error)


def _read_graph(transaction: neo4j.ManagedTransaction) -> tuple[list[Node], list[Relationship]]:
    # Store repeated IDs, labels, types, and keys once.
    strings: dict[str, str] = {}

    def share(text: str) -> str:
        return strings.setdefault(text, text)

    nodes = [
        Node(
            id=share(record["id"]),
            labels=tuple(map(share, record["labels"])),
            properties=_properties(record["properties"], share),
            embedding_dimensions={share(key): size for key, size in record["embedding_dimensions"]},
        )
        for record in transaction.run(_NODES_QUERY)
    ]
    relationships = [
        Relationship(
            id=record["id"],
            type=share(record["type"]),
            source_id=share(record["source_id"]),
            target_id=share(record["target_id"]),
            properties=_properties(record["properties"], share),
            embedding_dimensions={share(key): size for key, size in record["embedding_dimensions"]},
        )
        for record in transaction.run(_RELATIONSHIPS_QUERY)
    ]
    return nodes, relationships


def _properties(pairs: list[list[object]], share: Callable[[str], str]) -> dict[str, PropertyValue]:
    properties: dict[str, PropertyValue] = {}
    for key, value in pairs:
        converted = _convert(value)
        if converted is not None:
            properties[share(str(key))] = converted
    return properties


def _convert(value: object) -> PropertyValue | None:
    """Convert a driver value to a property value, or ``None`` for unsupported types."""
    if not isinstance(value, list):
        return _convert_scalar(value)
    scalars: list[Scalar] = []
    for item in value:
        scalar = _convert_scalar(item)
        if scalar is None:
            return None
        scalars.append(scalar)
    return scalars


def _convert_scalar(value: object) -> Scalar | None:
    if isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, neo4j.time.Date | neo4j.time.DateTime | neo4j.time.Time):
        try:
            return value.to_native()
        except ValueError:
            return None
    if isinstance(value, neo4j.time.Duration):
        return Duration(
            months=value.months,
            days=value.days,
            seconds=value.seconds,
            nanoseconds=value.nanoseconds,
        )
    if isinstance(value, neo4j.spatial.Point) and value.srid is not None:
        return Point(srid=value.srid, coordinates=tuple(value))
    return None
