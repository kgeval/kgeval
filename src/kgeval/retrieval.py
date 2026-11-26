"""Implement the retrievals that hold the element IDs a retriever returns for each test question."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import TypeAlias

from neo4j import Record
from neo4j.graph import Node, Relationship
from neo4j.graph import Path as GraphPath
from pydantic import TypeAdapter

from kgeval._yaml import load_yaml
from kgeval.errors import InputError
from kgeval.questions import QuestionSet

__all__ = ["Retrievals", "Retriever", "load_retrievals", "retrieve"]

logger = logging.getLogger(__name__)

Retriever: TypeAlias = Callable[[str], Iterable[object]]
"""A function that returns a question's element IDs, or driver nodes, relationships, and paths."""

Retrievals: TypeAlias = Mapping[str, tuple[str, ...]]
"""The element IDs retrieved for each question, by question text, in rank order and each once."""


def retrieve(questions: QuestionSet, retriever: Retriever) -> Retrievals:
    """Ask a retriever every question and collect the element IDs of what it returns.

    Args:
        questions: The questions to ask.
        retriever: Called with each question's text; returns, in rank order, element
            IDs, or nodes, relationships, and paths as the Neo4j driver returns them,
            alone or within records and lists.

    Returns:
        The retrievals, with each element ID once, at its first rank.

    Raises:
        InputError: If the retriever returns anything else, such as a property value.
    """
    logger.info("Retrieving for %d questions", len(questions.questions))
    return {
        question.text: _element_ids(question.text, retriever(question.text))
        for question in questions.questions
    }


def load_retrievals(path: str | Path) -> Retrievals:
    """Load retrievals from a YAML or JSON mapping of question text to element IDs.

    A JSON file looks like ``{"Who founded Acme?": ["4:9c1e:0", "5:9c1e:2"]}``.

    Args:
        path: The retrievals file.

    Returns:
        The retrievals, with each question's element IDs in the file's order, each once,
        at its first rank.

    Raises:
        InputError: If the file cannot be read or holds no such mapping.
    """
    loaded = load_yaml(path, TypeAdapter(dict[str, list[str]]))
    return {text: tuple(dict.fromkeys(ids)) for text, ids in loaded.items()}


def _element_ids(question: str, found: Iterable[object]) -> tuple[str, ...]:
    ids: dict[str, None] = {}
    for item in found:
        _collect(question, item, ids)
    return tuple(ids)


def _collect(question: str, item: object, ids: dict[str, None]) -> None:
    """Add the element IDs an item holds to ``ids``, in rank order, keeping first ranks."""
    if isinstance(item, str):
        ids.setdefault(item)
    elif isinstance(item, Node | Relationship):
        ids.setdefault(item.element_id)
    elif isinstance(item, GraphPath):
        # Rank a path's elements in traversal order: each relationship before its end node.
        steps = zip(item.relationships, item.nodes[1:], strict=True)
        for element in (item.start_node, *(element for step in steps for element in step)):
            ids.setdefault(element.element_id)
    elif isinstance(item, Record) or (isinstance(item, Iterable) and not isinstance(item, Mapping)):
        # Read records and lists, but not other mappings, which hold property values.
        for value in item.values() if isinstance(item, Record) else item:
            _collect(question, value, ids)
    else:
        raise InputError(
            f"the retriever returned a {type(item).__name__} for {question!r}; return "
            "element IDs, or nodes, relationships, and paths, alone or within records and lists"
        )
