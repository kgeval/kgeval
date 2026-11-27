"""Implement the helpers that show entities, facts, and chunk texts to a model."""

from __future__ import annotations

import re
from collections.abc import Iterable

from kgeval.graph.model import Node, Relationship
from kgeval.graph.names import normalize_name
from kgeval.graph.view import GraphView

__all__ = [
    "MAX_CHUNKS",
    "chunk_texts",
    "entity_name",
    "excerpt",
    "quote_chunks",
    "spell",
    "untagged",
]

MAX_CHUNKS = 3
"""The most chunks a model is shown for one entity or fact."""

# Match the tags that prompts quote graph text in, as loosely as a model reads them.
_TAG = re.compile(r"<\s*/?\s*(?:chunk|excerpt)\s*>", re.IGNORECASE)


def entity_name(graph: GraphView, entity: Node) -> str | None:
    """Return an entity's name when it has a letter or digit, which a model can look up.

    Args:
        graph: The graph the entity belongs to.
        entity: The entity.

    Returns:
        The name, or ``None`` when the entity has none with a letter or digit, as
        ``entity_uniqueness`` and the ground truth metrics also require.
    """
    name = graph.name(entity)
    return name if name and normalize_name(name) else None


def chunk_texts(graph: GraphView, chunk_ids: Iterable[str], names: Iterable[str]) -> list[str]:
    """Return the texts of up to ``MAX_CHUNKS`` chunks, preferring relevant ones.

    Texts that contain every name, ignoring case, come first; ties go to the
    alphabetically first text. Chunks without text are left out.

    Args:
        graph: The graph the chunks belong to.
        chunk_ids: The chunks' element IDs.
        names: The names a relevant text contains.

    Returns:
        The texts.
    """
    folded = [name.casefold() for name in names]
    texts = [text for chunk_id in chunk_ids if (text := graph.text(graph.node(chunk_id)))]
    # Order by content, not element ID, so that a rebuilt graph gets the same prompts.
    texts.sort(key=lambda text: (not all(name in text.casefold() for name in folded), text))
    return texts[:MAX_CHUNKS]


def excerpt(text: str, name: str, radius: int) -> str:
    """Return the part of a text around the first mention of a name, or its start.

    Args:
        text: A chunk's text.
        name: The name to find, ignoring case.
        radius: How many characters to keep on each side of the name.

    Returns:
        The excerpt with its whitespace collapsed, and ``…`` where text was cut.
    """
    start = max(text.casefold().find(name.casefold()), 0)
    begin = max(start - radius, 0)
    end = start + len(name) + radius
    kept = " ".join(text[begin:end].split())
    return untagged(("…" if begin else "") + kept + ("…" if end < len(text) else ""))


def quote_chunks(texts: Iterable[str]) -> str:
    """Return chunk texts as a model reads them, each between ``<chunk>`` tags.

    Args:
        texts: The chunk texts.

    Returns:
        The tagged texts, separated by blank lines.
    """
    return "\n\n".join(f"<chunk>\n{untagged(text)}\n</chunk>" for text in texts)


def untagged(text: str) -> str:
    """Return text from the graph with the tags that prompts quote text in escaped.

    Graph text comes from the documents the graph was built from, so none of it may end
    a quote, after which the rest of it could pass itself off as instructions.

    Args:
        text: Text from the graph, such as a name or a chunk's text.

    Returns:
        The text, with the angle brackets of every ``<chunk>`` and ``<excerpt>`` tag in it
        escaped.
    """
    return _TAG.sub(lambda tag: tag.group().replace("<", "&lt;").replace(">", "&gt;"), text)


def spell(graph: GraphView, element: Node | Relationship) -> str:
    """Return an entity or fact as a model reads it, with names in full.

    ``GraphView.describe`` shortens long names for display; a model needs them whole.

    Args:
        graph: The graph the element belongs to.
        element: An entity or a fact.

    Returns:
        For an entity, its quoted name, or ``unnamed``, followed by its node types in
        parentheses. For a fact, its source and target entities so spelled, joined by
        its relationship type in Cypher arrow notation.
    """
    if isinstance(element, Relationship):
        source = spell(graph, graph.node(element.source_id))
        target = spell(graph, graph.node(element.target_id))
        return untagged(f"{source} -[:{graph.relationship_type(element)}]-> {target}")
    name = graph.name(element)
    reference = f'"{name}"' if name else "unnamed"
    labels = ":".join(graph.layout.node_types(element))
    return untagged(f"{reference} ({labels})" if labels else reference)
