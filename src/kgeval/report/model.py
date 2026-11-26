"""Implement the report that records the results of one evaluation and how to reproduce them."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from kgeval.graph.layout import Layout, Role
from kgeval.graph.view import GraphView
from kgeval.metrics.base import MetricResult
from kgeval.report.markdown import DEFAULT_MAX_FINDINGS, render_markdown

__all__ = ["REPORT_FORMAT", "GraphSummary", "Report"]

REPORT_FORMAT = 2
"""The version of the report format that this kgeval writes and reads.

It grows whenever a report could hold what an older kgeval cannot read, so that older
releases reject newer reports with a clear message rather than misread them.
"""


class GraphSummary(BaseModel):
    """Count a graph's nodes and relationships, in total and by role.

    Attributes:
        nodes: All nodes.
        relationships: All relationships.
        entities: Entities.
        lexical_nodes: Lexical nodes.
        community_nodes: Community nodes.
        other_nodes: Nodes the layout ignores.
        facts: Relationships between two entities that are still current.
        expired_facts: Relationships between two entities that the layout marks as no
            longer current.
        provenance_relationships: Relationships between an entity and a lexical node.
        labels: Node count per label.
        relationship_types: Relationship count per relationship type.
    """

    model_config = ConfigDict(frozen=True)

    nodes: int
    relationships: int
    entities: int
    lexical_nodes: int
    community_nodes: int
    other_nodes: int
    facts: int
    expired_facts: int = 0
    provenance_relationships: int
    labels: dict[str, int]
    relationship_types: dict[str, int]

    @classmethod
    def of(cls, graph: GraphView) -> GraphSummary:
        """Summarize a graph.

        Args:
            graph: The graph to summarize.

        Returns:
            The summary, with label and type counts sorted by name.
        """
        roles = Counter(graph.role(node) for node in graph.graph.nodes)
        labels = Counter(label for node in graph.graph.nodes for label in node.labels)
        types = Counter(relationship.type for relationship in graph.graph.relationships)
        return cls(
            nodes=len(graph.graph.nodes),
            relationships=len(graph.graph.relationships),
            entities=roles[Role.ENTITY],
            lexical_nodes=roles[Role.LEXICAL],
            community_nodes=roles[Role.COMMUNITY],
            other_nodes=roles[Role.OTHER],
            facts=len(graph.facts),
            expired_facts=len(graph.expired_facts),
            provenance_relationships=len(graph.provenance),
            labels=dict(sorted(labels.items())),
            relationship_types=dict(sorted(types.items())),
        )


class Report(BaseModel):
    """Hold the results of one evaluation and everything needed to reproduce them.

    Rerunning the same kgeval version, layout, and metric signatures on a graph with the
    same fingerprint reproduces every deterministic result. Index a report by metric
    name to get that metric's result.

    Attributes:
        report_format: The version of the report format; reports without it have format
            1, the first.
        kgeval_version: The kgeval version that produced the report.
        graph_fingerprint: Identifies the exact graph content evaluated.
        layout: How the graph was read.
        summary: Counts of the graph's nodes and relationships.
        results: One result per metric, in evaluation order.
        metadata: Describes the run, such as the chunk size or prompt that built the
            graph. It annotates the report without changing any result.
    """

    model_config = ConfigDict(frozen=True)

    report_format: int = Field(default=1, ge=1, strict=True)
    kgeval_version: str
    graph_fingerprint: str
    layout: Layout
    summary: GraphSummary
    results: tuple[MetricResult, ...]
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _check_format(cls, data: object) -> object:
        # Reject other formats before their fields fail validation with obscure messages.
        if isinstance(data, Mapping):
            report_format = data.get("report_format", 1)
            if isinstance(report_format, int) and report_format > REPORT_FORMAT:
                raise ValueError(
                    f"the report has format {report_format}, which needs a newer kgeval; "
                    f"this one reads format {REPORT_FORMAT}"
                )
            if isinstance(report_format, int) and report_format < REPORT_FORMAT:
                raise ValueError(
                    f"the report has format {report_format}, from an older kgeval; this one "
                    f"reads format {REPORT_FORMAT}, so evaluate the graph again"
                )
        return data

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        """Return whether every metric with a threshold met it."""
        return all(result.passed is not False for result in self.results)

    def __getitem__(self, metric: str) -> MetricResult:
        """Return the result of the metric with the given name.

        Args:
            metric: The metric's name.

        Returns:
            The metric's result.

        Raises:
            KeyError: If no metric of that name was evaluated.
        """
        for result in self.results:
            if result.metric == metric:
                return result
        raise KeyError(metric)

    def to_markdown(self, max_findings: int = DEFAULT_MAX_FINDINGS) -> str:
        """Render the graph summary, score tables, and findings as Markdown.

        Text taken from the graph is escaped, so it cannot inject Markdown.

        Args:
            max_findings: How many findings to list per metric; the rest are counted.

        Returns:
            The Markdown text.
        """
        return render_markdown(self, max_findings)

    def _repr_markdown_(self) -> str:
        # Render reports as Markdown in notebooks.
        return self.to_markdown()
