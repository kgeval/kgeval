"""Assign each node a role: entity, lexical node, community node, or other."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, field_serializer

from kgeval.graph.model import Graph, Node

__all__ = ["BASE_LABELS", "COMMUNITY_LABELS", "LEXICAL_LABELS", "Layout", "Role"]

LEXICAL_LABELS = frozenset({"Chunk", "Document", "Episodic", "__Chunk__", "__Document__"})
"""Labels that well-known builders give to lexical nodes.

A graph's nodes with these labels are lexical nodes only when they hold text, or are
joined to lexical nodes that do, as a document is to its chunks, so that a ``Document``
node type of a domain graph stays a node type.
"""

COMMUNITY_LABELS = frozenset({"__Community__"})
"""Labels that well-known builders give to community nodes.

Graphiti's ``Community`` counts too, in graphs with Graphiti's ``Episodic`` label, so
that a ``Community`` node type of another graph stays a node type.
"""

BASE_LABELS = frozenset({"__Entity__", "__KGBuilder__", "__Node__"})
"""Base labels that well-known builders add to every entity, next to its node type.

Some builders also put ``__KGBuilder__`` or ``__Node__`` on chunks; lexical labels take
precedence, so those chunks stay lexical nodes. Graphiti's ``Entity`` counts too, in
graphs with Graphiti's ``Episodic`` label.
"""


class Role(StrEnum):
    """Name the part a node plays in a knowledge graph."""

    ENTITY = "entity"
    LEXICAL = "lexical"
    COMMUNITY = "community"
    OTHER = "other"


class Layout(BaseModel):
    """Describe which labels give each role, and which properties hold names, text, and types.

    A node's role is decided by its labels, with this precedence: lexical, community,
    entity. When ``base_labels`` is empty, every remaining node is an entity;
    otherwise only nodes with a base label are, and the rest are ignored as
    ``Role.OTHER``.

    Attributes:
        lexical_labels: Labels of lexical nodes.
        community_labels: Labels of community nodes.
        base_labels: Labels that mark a node as an entity without naming its node type.
        name_keys: Property keys tried, in order, for a node's name.
        text_keys: Property keys tried, in order, for a lexical node's text.
        type_keys: Property keys tried, in order, for a fact's relationship type, for
            builders that store every fact under one generic relationship type, such
            as Graphiti's ``RELATES_TO``, which keeps the type in ``name``. A fact
            without a non-blank string under any of them keeps its own relationship
            type.
        description_keys: Property keys tried, in order, for an entity's or fact's
            description, such as Microsoft GraphRAG's ``description`` and Graphiti's
            ``fact``, the statement a fact makes in words, and ``summary``, its
            description of an entity.
        expired_keys: Property keys that mark a relationship between entities as no
            longer current when it has any of them, such as Graphiti's ``expired_at``,
            set on facts that later information invalidated. Such relationships are
            the graph's history, not its facts.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    lexical_labels: frozenset[str] = frozenset()
    community_labels: frozenset[str] = frozenset()
    base_labels: frozenset[str] = frozenset()
    name_keys: tuple[str, ...] = ("name", "title")
    text_keys: tuple[str, ...] = ("text", "content")
    type_keys: tuple[str, ...] = ()
    description_keys: tuple[str, ...] = ("fact", "description", "summary")
    expired_keys: tuple[str, ...] = ()

    @field_serializer("lexical_labels", "community_labels", "base_labels")
    def _serialize_sorted(self, labels: frozenset[str]) -> list[str]:
        # Sort sets so serialized reports are reproducible byte for byte.
        return sorted(labels)

    @classmethod
    def detect(cls, graph: Graph, overrides: Layout | None = None) -> Layout:
        """Return the layout implied by the well-known labels and properties of a graph.

        Args:
            graph: The graph to inspect.
            overrides: A layout whose explicitly set fields replace the detected ones;
                its text and name keys also guide detection.

        Returns:
            A layout using whichever of ``LEXICAL_LABELS``, ``COMMUNITY_LABELS``, and
            ``BASE_LABELS`` the graph shows, as their docstrings describe, with the
            default text and description keys. In a graph with Graphiti's
            ``Episodic`` label, ``Entity`` is a base label and ``Community`` a
            community label too, the type keys are ``name``, where Graphiti keeps the
            relationship types of its ``RELATES_TO`` facts, and the expired keys are
            ``expired_at``. The name keys are ``name`` and ``title``, followed by
            ``id`` when the graph has the ``__Entity__`` label or no entity has a name
            or title: LLM Graph Builder and LangChain store entity names in ``id``.
        """
        given = cls() if overrides is None else overrides
        fixed = set() if overrides is None else overrides.model_fields_set
        present = frozenset(label for node in graph.nodes for label in node.labels)
        graphiti = "Episodic" in present
        detected: dict[str, object] = {
            "lexical_labels": _lexical_labels(graph, present & LEXICAL_LABELS, given.text_keys),
            "community_labels": present
            & ((COMMUNITY_LABELS | {"Community"}) if graphiti else COMMUNITY_LABELS),
            "base_labels": present & ((BASE_LABELS | {"Entity"}) if graphiti else BASE_LABELS),
            "type_keys": ("name",) if graphiti else (),
            "expired_keys": ("expired_at",) if graphiti else (),
        }
        layout = given.model_copy(
            update={field: value for field, value in detected.items() if field not in fixed}
        )
        if "name_keys" in fixed:
            return layout
        named = any(
            _has_text(node, layout.name_keys)
            for node in graph.nodes
            if layout.role(node) is Role.ENTITY
        )
        if "__Entity__" in present or not named:
            return layout.model_copy(update={"name_keys": (*layout.name_keys, "id")})
        return layout

    def role(self, node: Node) -> Role:
        """Return the role a node plays under this layout.

        Args:
            node: The node to classify.

        Returns:
            The node's role.
        """
        labels = set(node.labels)
        if labels & self.lexical_labels:
            return Role.LEXICAL
        if labels & self.community_labels:
            return Role.COMMUNITY
        if not self.base_labels or labels & self.base_labels:
            return Role.ENTITY
        return Role.OTHER

    def node_types(self, node: Node) -> tuple[str, ...]:
        """Return a node's node types: its labels other than base labels.

        Args:
            node: The node whose labels to filter.

        Returns:
            The node's labels minus ``base_labels``, sorted.
        """
        return tuple(label for label in node.labels if label not in self.base_labels)


def _lexical_labels(
    graph: Graph, candidates: frozenset[str], text_keys: tuple[str, ...]
) -> frozenset[str]:
    """Return the candidate labels of nodes that hold text, or that are joined to such nodes."""
    with_text = {
        node.id
        for node in graph.nodes
        if candidates.intersection(node.labels) and _has_text(node, text_keys)
    }
    lexical = with_text | {
        other_id
        for relationship in graph.relationships
        for node_id, other_id in (
            (relationship.source_id, relationship.target_id),
            (relationship.target_id, relationship.source_id),
        )
        if node_id in with_text
    }
    return frozenset(
        label
        for node in graph.nodes
        if node.id in lexical
        for label in node.labels
        if label in candidates
    )


def _has_text(node: Node, keys: tuple[str, ...]) -> bool:
    return any(isinstance(value := node.properties.get(key), str) and value.strip() for key in keys)
