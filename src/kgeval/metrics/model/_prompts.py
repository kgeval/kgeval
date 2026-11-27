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
"""Limit how many chunks a model is shown for one entity or fact."""

# Match the tags that prompts quote graph text in, as loosely as a model reads them.
_TAG = re.compile(r"<\s*/?\s*(?:chunk|excerpt)\s*>", re.IGNORECASE)


def entity_name(graph: GraphView, entity: Node) -> str | None:
    """Return an entity's name when it has a letter or digit, which a model can look up."""
    name = graph.name(entity)
    return name if name and normalize_name(name) else None


def chunk_texts(graph: GraphView, chunk_ids: Iterable[str], names: Iterable[str]) -> list[str]:
    """Return up to ``MAX_CHUNKS`` chunk texts, those that contain every name first."""
    folded = [name.casefold() for name in names]
    texts = [text for chunk_id in chunk_ids if (text := graph.text(graph.node(chunk_id)))]
    # Order by content, not element ID, so that a rebuilt graph gets the same prompts.
    texts.sort(key=lambda text: (not all(name in text.casefold() for name in folded), text))
    return texts[:MAX_CHUNKS]


def excerpt(text: str, name: str, radius: int) -> str:
    """Return the text around a name's first mention, or its start, marking each cut."""
    start = max(text.casefold().find(name.casefold()), 0)
    begin = max(start - radius, 0)
    end = start + len(name) + radius
    kept = " ".join(text[begin:end].split())
    return untagged(("…" if begin else "") + kept + ("…" if end < len(text) else ""))


def quote_chunks(texts: Iterable[str]) -> str:
    """Return chunk texts as a model reads them, each between ``<chunk>`` tags."""
    return "\n\n".join(f"<chunk>\n{untagged(text)}\n</chunk>" for text in texts)


def untagged(text: str) -> str:
    """Escape the prompts' quoting tags in graph text, so that no graph text can end a quote."""
    return _TAG.sub(lambda tag: tag.group().replace("<", "&lt;").replace(">", "&gt;"), text)


def spell(graph: GraphView, element: Node | Relationship) -> str:
    """Return an entity or fact as a model reads it, with names in full, unlike ``describe``."""
    if isinstance(element, Relationship):
        source = spell(graph, graph.node(element.source_id))
        target = spell(graph, graph.node(element.target_id))
        return untagged(f"{source} -[:{graph.relationship_type(element)}]-> {target}")
    name = graph.name(element)
    reference = f'"{name}"' if name else "unnamed"
    labels = ":".join(graph.layout.node_types(element))
    return untagged(f"{reference} ({labels})" if labels else reference)
