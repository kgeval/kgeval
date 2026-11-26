"""Implement the helpers that collect a graph's vocabulary and report duplicate terms."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence

from kgeval.graph.model import Node, Relationship
from kgeval.graph.view import GraphView
from kgeval.metrics.base import Finding

__all__ = ["enumerate_terms", "merge_findings", "vocabulary"]


def vocabulary(graph: GraphView) -> dict[str, dict[str, list[Node | Relationship]]]:
    """Map each kind of term to the terms of that kind in use and the elements using them.

    Args:
        graph: The graph whose vocabulary to collect.

    Returns:
        The node types, relationship types, and property keys, under the keys ``"node
        type"``, ``"relationship type"``, and ``"property key"``, each mapped to the
        entities or facts that use it, in ID order.
    """
    node_types: defaultdict[str, list[Node | Relationship]] = defaultdict(list)
    relationship_types: defaultdict[str, list[Node | Relationship]] = defaultdict(list)
    property_keys: defaultdict[str, list[Node | Relationship]] = defaultdict(list)
    for entity in graph.entities:
        for node_type in graph.layout.node_types(entity):
            node_types[node_type].append(entity)
        for key in entity.properties:
            property_keys[key].append(entity)
    for fact in graph.facts:
        relationship_types[graph.relationship_type(fact)].append(fact)
        for key in fact.properties:
            property_keys[key].append(fact)
    return {
        "node type": dict(node_types),
        "relationship type": dict(relationship_types),
        "property key": dict(property_keys),
    }


def merge_findings(
    kind: str,
    groups: Iterable[Sequence[str]],
    carriers: Mapping[str, Sequence[Node | Relationship]],
    message: Callable[[list[str], str], str],
) -> tuple[int, list[Finding]]:
    """Report every group of terms that mean the same, keeping the most used term of each.

    In each group, the term most elements use counts as intended, ties going to the
    alphabetically first.

    Args:
        kind: The kind of the terms, such as ``"node type"``.
        groups: The groups of terms; groups of fewer than two terms are ignored.
        carriers: Maps each term to the elements that use it.
        message: Builds a finding's message from the other terms and the intended one.

    Returns:
        How many terms duplicate another, and one finding per group, about the intended
        term and listing the elements that use the other terms.
    """
    failed = 0
    findings: list[Finding] = []
    for group in groups:
        if len(group) < 2:
            continue
        intended = min(group, key=lambda term: (-len(carriers[term]), term))
        others = sorted(term for term in group if term != intended)
        failed += len(others)
        elements = [element for term in others for element in carriers[term]]
        findings.append(
            Finding.about(elements, message(others, intended), subject=f"{kind} {intended}")
        )
    return failed, findings


def enumerate_terms(terms: Sequence[str], conjunction: str = "and") -> str:
    """Join terms into a list in words, such as ``"A, B, and C"``.

    Args:
        terms: At least one term.
        conjunction: The word before the last term, such as ``"or"``.

    Returns:
        The list.
    """
    if len(terms) <= 2:
        return f" {conjunction} ".join(terms)
    return f"{', '.join(terms[:-1])}, {conjunction} {terms[-1]}"
