"""Implement the vocabulary uniqueness metric."""

from __future__ import annotations

from collections import defaultdict
from functools import partial
from typing import ClassVar

from kgeval.graph.names import normalize_type
from kgeval.graph.view import GraphView
from kgeval.metrics._vocabulary import enumerate_terms, merge_findings, vocabulary
from kgeval.metrics.base import Finding, Metric, MetricCategory, MetricResult

__all__ = ["VocabularyUniqueness"]


class VocabularyUniqueness(Metric):
    """Measure the share of vocabulary terms in use that are distinct, counting variants once.

    The vocabulary is the node types and property keys of the entities, and the
    relationship types and property keys of the facts. Terms of one kind are spelling
    variants when they differ only in case, separators, camel case or plural form, as
    ``normalize_type`` defines. In each group of variants, the one most elements use
    counts as intended, ties going to the alphabetically first, and ``failed`` counts
    the others; each finding lists the elements that use them. ``vocabulary_resolution``
    also finds terms that mean the same in different words.
    """

    name: ClassVar[str] = "vocabulary_uniqueness"
    category: ClassVar[MetricCategory] = MetricCategory.INTRINSIC
    unit: ClassVar[str] = "vocabulary terms"

    def evaluate(self, graph: GraphView) -> MetricResult:
        """Group each kind of term by normalized spelling and report every group of two or more.

        Args:
            graph: The graph to evaluate.

        Returns:
            The metric's result.
        """
        evaluated = failed = 0
        findings: list[Finding] = []
        for kind, carriers in vocabulary(graph).items():
            groups: defaultdict[str, list[str]] = defaultdict(list)
            for term in sorted(carriers):
                groups[normalize_type(term)].append(term)
            kind_failed, kind_findings = merge_findings(
                kind, (groups[key] for key in sorted(groups)), carriers, partial(_variants, kind)
            )
            evaluated += len(carriers)
            failed += kind_failed
            findings += kind_findings
        return self.result(evaluated, failed, findings)


def _variants(kind: str, others: list[str], intended: str) -> str:
    verb = "spells" if len(others) == 1 else "spell"
    return f"{enumerate_terms(others)} {verb} the {kind} {intended} differently"
