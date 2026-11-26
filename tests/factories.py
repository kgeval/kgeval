"""Implement the factories of test graphs, and the comparison of graphs by content."""

from __future__ import annotations

from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from kgeval.graph.layout import Layout
from kgeval.graph.model import Graph, Node, PropertyValue, Relationship, canonical_json
from kgeval.graph.view import GraphView
from kgeval.schema.model import GraphSchema

EXAMPLE_DIR = Path(__file__).parents[1] / "examples" / "singapore"

EXAMPLE_SCHEMA: dict[str, Any] = yaml.safe_load(
    (EXAMPLE_DIR / "schema.yaml").read_text(encoding="utf-8")
)

# Miss the birthplace in both questions that need it, and add Kwa Geok Choo to the birth date.
EXAMPLE_RETRIEVALS: dict[str, tuple[str, ...]] = {
    "Who founded the People's Action Party?": ("chunk1", "founded_pap", "lee", "pap"),
    "Where was Lee Kuan Yew born?": ("singapore",),
    "What did Lee Kuan Yew co-found?": ("founded_pap",),
    "When was Lee Kuan Yew born?": ("kwa", "lee"),
    "In which country was the founder of the People's Action Party born?": (
        "founded_pap",
        "lee",
        "pap",
    ),
}


def node(
    node_id: str,
    *labels: str,
    embeddings: dict[str, int] | None = None,
    **properties: PropertyValue,
) -> Node:
    return Node(
        id=node_id, labels=labels, properties=properties, embedding_dimensions=embeddings or {}
    )


def relationship(
    relationship_id: str,
    source_id: str,
    type_: str,
    target_id: str,
    embeddings: dict[str, int] | None = None,
    **properties: PropertyValue,
) -> Relationship:
    return Relationship(
        id=relationship_id,
        type=type_,
        source_id=source_id,
        target_id=target_id,
        properties=properties,
        embedding_dimensions=embeddings or {},
    )


def graph(*elements: Node | Relationship) -> Graph:
    return Graph(
        nodes=tuple(element for element in elements if isinstance(element, Node)),
        relationships=tuple(element for element in elements if isinstance(element, Relationship)),
    )


def view(*elements: Node | Relationship, layout: Layout | None = None) -> GraphView:
    return GraphView(graph(*elements), layout)


def content(graph: Graph) -> tuple[Counter[str], Counter[tuple[str, str, str, str]]]:
    """Return a graph's nodes and relationships by content, without element IDs."""
    nodes = {
        node.id: canonical_json([node.labels, node.properties, node.embedding_dimensions])
        for node in graph.nodes
    }
    relationships = Counter(
        (
            nodes[link.source_id],
            link.type,
            nodes[link.target_id],
            canonical_json([link.properties, link.embedding_dimensions]),
        )
        for link in graph.relationships
    )
    return Counter(nodes.values()), relationships


def example_graph() -> Graph:
    """Return the graph that examples/singapore/seed.cypher creates, with readable IDs.

    Node IDs are the seed's variable names; embeddings keep only their dimensions, as
    sources do.
    """
    born = date(1923, 9, 16)
    return graph(
        node("document", "Document", fileName="singapore.txt"),
        node(
            "chunk1",
            "Chunk",
            embeddings={"embedding": 2},
            text="Lee Kuan Yew was born in Singapore. He co-founded the People's Action Party "
            "and the law firm Lee & Lee.",
        ),
        node(
            "chunk2",
            "Chunk",
            embeddings={"embedding": 2},
            text="Kwa Geok Choo married Lee Kuan Yew.",
        ),
        node("chunk3", "Chunk", text="Table of contents"),
        relationship("part_of_1", "chunk1", "PART_OF", "document"),
        relationship("part_of_2", "chunk2", "PART_OF", "document"),
        relationship("part_of_3", "chunk3", "PART_OF", "document"),
        relationship("next_chunk_1", "chunk1", "NEXT_CHUNK", "chunk2"),
        relationship("next_chunk_2", "chunk2", "NEXT_CHUNK", "chunk3"),
        node("lee", "__Entity__", "Person", id="Lee Kuan Yew", born=born),
        node("singapore", "__Entity__", "Location", id="Singapore"),
        node("pap", "__Entity__", "Organization", id="People's Action Party"),
        node("unnamed", "__Entity__", "Organization"),
        node("kwa", "__Entity__", "Person", id="Kwa Geok Choo", born="1920"),
        node("duplicate", "__Entity__", "Person", id="lee kuan yew", born=born),
        node("marina_bay_sands", "__Entity__", "Landmark", id="Marina Bay Sands"),
        node("he", "__Entity__", "PERSON", id="He"),
        relationship("has_entity_1", "chunk1", "HAS_ENTITY", "lee"),
        relationship("has_entity_2", "chunk1", "HAS_ENTITY", "singapore"),
        relationship("has_entity_3", "chunk1", "HAS_ENTITY", "pap"),
        relationship("has_entity_4", "chunk1", "HAS_ENTITY", "unnamed"),
        relationship("has_entity_5", "chunk1", "HAS_ENTITY", "he"),
        relationship("has_entity_6", "chunk2", "HAS_ENTITY", "lee"),
        relationship("has_entity_7", "chunk2", "HAS_ENTITY", "kwa"),
        relationship("has_entity_8", "chunk2", "HAS_ENTITY", "duplicate"),
        relationship("born_in", "lee", "BORN_IN", "singapore"),
        relationship("born_in_copy", "lee", "BORN_IN", "singapore"),
        relationship("founded_pap", "lee", "FOUNDED", "pap"),
        relationship("founded_unnamed", "lee", "FOUNDED", "unnamed"),
        relationship("duplicate_founded", "duplicate", "FOUNDED", "pap"),
        relationship("he_founded", "he", "Founded", "pap"),
        relationship("married", "kwa", "MARRIED", "lee"),
        relationship("knows", "kwa", "KNOWS", "kwa"),
        node("document2", "Document", fileName="early-years.txt"),
        node(
            "chunk4",
            "Chunk",
            embeddings={"embedding": 2},
            text="Lee studied law at Fitzwilliam College, Cambridge. As a boy, Lee Kuan Yew "
            "attended Raffles Institution.",
        ),
        node(
            "chunk5",
            "Chunk",
            embeddings={"embedding": 2},
            text="Raffles founded modern Singapore in 1819. Lee Kuan Yew became its first prime "
            "minister in 1959.",
        ),
        relationship("part_of_4", "chunk4", "PART_OF", "document2"),
        relationship("part_of_5", "chunk5", "PART_OF", "document2"),
        relationship("next_chunk_3", "chunk4", "NEXT_CHUNK", "chunk5"),
        node("surname", "__Entity__", "Person", id="Lee"),
        node("fitzwilliam", "__Entity__", "Person", id="Fitzwilliam College"),
        node("raffles", "__Entity__", "Organization", id="Raffles"),
        relationship("has_entity_9", "chunk4", "HAS_ENTITY", "lee"),
        relationship("has_entity_10", "chunk4", "HAS_ENTITY", "surname"),
        relationship("has_entity_11", "chunk4", "HAS_ENTITY", "fitzwilliam"),
        relationship("has_entity_12", "chunk4", "HAS_ENTITY", "raffles"),
        relationship("has_entity_13", "chunk5", "HAS_ENTITY", "raffles"),
        relationship("has_entity_14", "chunk5", "HAS_ENTITY", "singapore"),
        relationship("studied_at", "surname", "STUDIED_AT", "fitzwilliam"),
        relationship("attended", "lee", "ATTENDED", "raffles"),
        relationship("raffles_founded", "raffles", "FOUNDED", "singapore"),
        node(
            "community",
            "__Community__",
            id="0-0",
            level=0,
            title="Lee Kuan Yew and the People's Action Party",
        ),
        *(
            relationship(f"in_community_{member}", member, "IN_COMMUNITY", "community")
            for member in ("lee", "singapore", "pap", "unnamed", "kwa", "duplicate", "he")
        ),
    )


def example_view() -> GraphView:
    return GraphView(example_graph())


def example_schema(**changes: Any) -> GraphSchema:
    """Return the example schema with the given top-level fields replaced."""
    return GraphSchema.model_validate(EXAMPLE_SCHEMA | changes)
