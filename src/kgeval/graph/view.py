"""Implement the view that indexes a graph's entities, facts, and chunks."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

from kgeval.graph.layout import Layout, Role
from kgeval.graph.model import Graph, Node, Relationship

__all__ = ["GraphView"]

_PREVIEW_LENGTH = 60


class GraphView:
    """Index a graph by role so metrics can ask about entities, facts, and chunks.

    A fact is a relationship whose source and target nodes are both entities, unless it
    has one of the layout's expired keys, which makes it an expired fact that metrics
    do not evaluate. A provenance relationship joins an entity and a lexical node, in
    either direction; that lexical node is a chunk the entity is grounded in, and
    ``chunk_ids`` returns them. All other relationships are neither facts nor
    provenance relationships.

    Args:
        graph: The graph to index.
        layout: How to read the graph; detected from its labels when omitted.

    Attributes:
        graph: The underlying graph.
        layout: The layout in use.
        entities: Entity nodes, sorted by ID.
        lexical_nodes: Lexical nodes, sorted by ID.
        chunks: Lexical nodes that share a lexical label with a node some entity is
            grounded in, sorted by ID. In a lexical graph of documents and chunks, these
            are every chunk, whether or not it grounds an entity, and no document.
        community_nodes: Community nodes, sorted by ID.
        facts: Facts, sorted by ID.
        expired_facts: Relationships between entities that the layout's expired keys
            mark as no longer current, sorted by ID.
        provenance: Provenance relationships, sorted by ID.
    """

    def __init__(self, graph: Graph, layout: Layout | None = None) -> None:
        self.graph = graph
        self.layout = layout if layout is not None else Layout.detect(graph)
        self._nodes = {node.id: node for node in graph.nodes}
        self._roles = {node.id: self.layout.role(node) for node in graph.nodes}
        self.entities = self._with_role(Role.ENTITY)
        self.lexical_nodes = self._with_role(Role.LEXICAL)
        self.community_nodes = self._with_role(Role.COMMUNITY)
        facts: list[Relationship] = []
        expired_facts: list[Relationship] = []
        provenance: list[Relationship] = []
        chunk_ids: defaultdict[str, set[str]] = defaultdict(set)
        for relationship in graph.relationships:
            source_role = self._roles[relationship.source_id]
            target_role = self._roles[relationship.target_id]
            if source_role is Role.ENTITY and target_role is Role.ENTITY:
                expired = any(key in relationship.properties for key in self.layout.expired_keys)
                (expired_facts if expired else facts).append(relationship)
                continue
            if source_role is Role.ENTITY and target_role is Role.LEXICAL:
                entity_id, chunk_id = relationship.source_id, relationship.target_id
            elif source_role is Role.LEXICAL and target_role is Role.ENTITY:
                entity_id, chunk_id = relationship.target_id, relationship.source_id
            else:
                continue
            provenance.append(relationship)
            chunk_ids[entity_id].add(chunk_id)
        self.facts = tuple(facts)
        self.expired_facts = tuple(expired_facts)
        self.provenance = tuple(provenance)
        self._chunk_ids = {key: frozenset(ids) for key, ids in chunk_ids.items()}
        # Remember node descriptions, since findings and fact descriptions repeat them.
        self._descriptions: dict[str, str] = {}
        grounding = {chunk_id for ids in chunk_ids.values() for chunk_id in ids}
        examples = grounding or {node.id for node in self.lexical_nodes if self.text(node)}
        chunk_labels = self.layout.lexical_labels.intersection(
            label for node_id in examples for label in self._nodes[node_id].labels
        )
        self.chunks = tuple(
            node for node in self.lexical_nodes if chunk_labels.intersection(node.labels)
        )

    def node(self, node_id: str) -> Node:
        """Return the node with the given ID.

        Args:
            node_id: The node's element ID.

        Returns:
            The node.

        Raises:
            KeyError: If the graph has no such node.
        """
        return self._nodes[node_id]

    def role(self, node: Node) -> Role:
        """Return the role a node plays in this graph.

        Args:
            node: A node of this graph.

        Returns:
            The node's role.
        """
        return self._roles[node.id]

    def chunk_ids(self, entity_id: str) -> frozenset[str]:
        """Return the IDs of the chunks an entity is grounded in.

        Args:
            entity_id: The entity's element ID.

        Returns:
            The chunk IDs; empty when the entity has no provenance relationship.
        """
        return self._chunk_ids.get(entity_id, frozenset())

    def fact_chunk_ids(self, fact: Relationship) -> frozenset[str]:
        """Return the IDs of the chunks that ground both of a fact's entities.

        Args:
            fact: The fact.

        Returns:
            The chunk IDs; empty when no chunk grounds both its source and its target.
        """
        return self.chunk_ids(fact.source_id) & self.chunk_ids(fact.target_id)

    def chunks_of(self, documents: Iterable[Mapping[str, str]]) -> frozenset[str]:
        """Return the IDs of the chunks that make up the selected documents.

        A lexical node is selected when it has every property of one selector, with the
        selector's value. The chunks are the selected chunks, and the chunks that a
        relationship joins to a selected lexical node that is not a chunk, such as a
        document node.

        Args:
            documents: The selectors, each mapping property keys to values.

        Returns:
            The chunk IDs.
        """
        selectors = list(documents)
        selected = {
            node.id
            for node in self.lexical_nodes
            if any(
                all(node.properties.get(key) == value for key, value in selector.items())
                for selector in selectors
            )
        }
        chunk_ids = {chunk.id for chunk in self.chunks}
        containers = selected - chunk_ids
        joined = {
            other_id
            for relationship in self.graph.relationships
            for node_id, other_id in (
                (relationship.source_id, relationship.target_id),
                (relationship.target_id, relationship.source_id),
            )
            if node_id in containers and other_id in chunk_ids
        }
        return frozenset((selected & chunk_ids) | joined)

    def name(self, node: Node) -> str | None:
        """Return the first non-blank string a node holds under the layout's name keys.

        Args:
            node: The node to name.

        Returns:
            The stripped name, or ``None`` when the node has none.
        """
        return _first_text(node, self.layout.name_keys)

    def text(self, node: Node) -> str | None:
        """Return the first non-blank string a node holds under the layout's text keys.

        Args:
            node: The node, usually a lexical one.

        Returns:
            The stripped text, or ``None`` when the node has none.
        """
        return _first_text(node, self.layout.text_keys)

    def relationship_type(self, relationship: Relationship) -> str:
        """Return a relationship's type, read from the layout's type keys when it has one there.

        Args:
            relationship: The relationship, usually a fact.

        Returns:
            The first non-blank string among the layout's type keys, stripped, or else
            the relationship's own type.
        """
        return _first_text(relationship, self.layout.type_keys) or relationship.type

    def description(self, element: Node | Relationship) -> str | None:
        """Return an entity's or fact's description, read from the layout's description keys.

        Args:
            element: The node or relationship, usually an entity or a fact.

        Returns:
            The first non-blank string among the layout's description keys, stripped,
            or ``None`` when the element has none.
        """
        return _first_text(element, self.layout.description_keys)

    def describe(self, element: Node | Relationship) -> str:
        """Return a short human-readable reference to a node or relationship.

        References name elements by content, never by element ID, so they are the same
        in every run on the same graph content.

        Args:
            element: The node or relationship to describe.

        Returns:
            A node's quoted name, else a quoted preview of its text, else ``unnamed``,
            followed by its node types in parentheses. A relationship's source and
            target nodes, each described as a node, joined by its relationship type in
            Cypher arrow notation.
        """
        if isinstance(element, Relationship):
            source = self.describe(self.node(element.source_id))
            target = self.describe(self.node(element.target_id))
            return f"{source} -[:{self.relationship_type(element)}]-> {target}"
        if (description := self._descriptions.get(element.id)) is None:
            content = self.name(element) or self.text(element)
            reference = _quote(content) if content else "unnamed"
            labels = ":".join(self.layout.node_types(element))
            description = f"{reference} ({labels})" if labels else reference
            self._descriptions[element.id] = description
        return description

    def _with_role(self, role: Role) -> tuple[Node, ...]:
        return tuple(node for node in self.graph.nodes if self._roles[node.id] is role)


def _first_text(element: Node | Relationship, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = element.properties.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _quote(text: str) -> str:
    flat = " ".join(text.split())
    if len(flat) > _PREVIEW_LENGTH:
        flat = flat[: _PREVIEW_LENGTH - 1] + "…"
    return f'"{flat}"'
