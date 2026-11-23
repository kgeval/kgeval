"""Render reports and comparisons as Markdown."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from kgeval.graph.layout import Layout
from kgeval.metrics.base import Finding, MetricCategory, MetricResult

if TYPE_CHECKING:
    from kgeval.report.comparison import Comparison, MetricChange
    from kgeval.report.model import Report

__all__ = ["DEFAULT_MAX_FINDINGS", "render_comparison_markdown", "render_markdown"]

DEFAULT_MAX_FINDINGS = 10
"""The default for how many findings Markdown lists per metric."""

_MAX_IDS_SHOWN = 10

_CATEGORY_TITLES = {
    MetricCategory.INTRINSIC: "Intrinsic graph metrics (deterministic)",
    MetricCategory.SCHEMA: "Schema graph metrics (deterministic)",
    MetricCategory.REFERENCE: "Reference graph metrics (deterministic)",
    MetricCategory.JUDGED: "Judged graph metrics (LLM judge)",
    MetricCategory.RETRIEVAL: "Retrieval metrics (deterministic)",
}

# Match Markdown syntax characters, except underscores inside words, which cannot open emphasis.
_MARKDOWN_SYNTAX = re.compile(r"[\\`*\[<~|]|_(?![^\W_])|(?<![^\W_])_")


def render_markdown(report: Report, max_findings: int = DEFAULT_MAX_FINDINGS) -> str:
    """Render a report as Markdown; see ``Report.to_markdown``.

    Args:
        report: The report to render.
        max_findings: How many findings to list per metric; the rest are counted.

    Returns:
        The Markdown text.
    """
    summary = report.summary
    lines = ["# kgeval report", ""]
    if any(result.passed is not None for result in report.results):
        lines += [f"**{'PASSED' if report.passed else 'FAILED'}**", ""]
    nodes = ", ".join(
        [
            _count(summary.entities, "entity", "entities"),
            _count(summary.lexical_nodes, "lexical node"),
            _count(summary.community_nodes, "community node"),
            _count(summary.other_nodes, "other node"),
        ]
    )
    relationships = ", ".join(
        [
            _count(summary.facts, "fact"),
            *([_count(summary.expired_facts, "expired fact")] if summary.expired_facts else []),
            _count(summary.provenance_relationships, "provenance relationship"),
        ]
    )
    if report.metadata:
        lines.append(f"- Metadata: {_describe_metadata(report.metadata)}")
    lines += [
        f"- Graph: {_count(summary.nodes, 'node')} ({nodes}) and "
        f"{_count(summary.relationships, 'relationship')} ({relationships})",
        f"- Layout: {_describe_layout(report.layout)}",
        f"- Reproduce with kgeval {report.kgeval_version} on graph `{report.graph_fingerprint}`",
        "",
        "## Scores",
    ]
    for category, title in _CATEGORY_TITLES.items():
        results = [result for result in report.results if result.category is category]
        if results:
            lines += [
                "",
                f"### {title}",
                "",
                "| Metric | Score | Failed | Threshold | Result |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
            lines += [_score_row(result) for result in results]
    with_findings = [result for result in report.results if result.findings]
    if with_findings:
        lines += ["", "## Findings"]
    for result in with_findings:
        counts = f"{result.failed:,} of {result.evaluated:,} {result.unit} failed"
        lines += ["", f"### {result.metric}: {counts}", ""]
        lines += _finding_items(result.findings, max_findings)
    return "\n".join(lines) + "\n"


def render_comparison_markdown(
    comparison: Comparison, max_findings: int = DEFAULT_MAX_FINDINGS
) -> str:
    """Render a comparison as Markdown; see ``Comparison.to_markdown``.

    Args:
        comparison: The comparison to render.
        max_findings: How many new and resolved findings to list per metric; the rest
            are counted.

    Returns:
        The Markdown text.
    """
    lines = [
        "# kgeval comparison",
        "",
        f"**{'REGRESSED' if comparison.regressed else 'NO REGRESSION'}**",
        "",
        f"- Base: {_describe_run(comparison.base)}",
        f"- Head: {_describe_run(comparison.head)}",
        "",
        "## Summary",
        "",
        f"Scores: {_describe_trends(comparison)}",
    ]
    lines += _caveats(comparison.base, comparison.head)
    lines += _metadata_changes(comparison.base.metadata, comparison.head.metadata)
    lines += [
        "",
        "## Scores",
        "",
        "| Metric | Base | Head | Change | New | Resolved | Note |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    lines += [_change_row(change) for change in comparison.changes]
    new = {change.metric: change.new_findings for change in comparison.changes}
    resolved = {change.metric: change.resolved_findings for change in comparison.changes}
    lines += _changed_findings("New findings", "new", new, max_findings)
    lines += _changed_findings("Resolved findings", "resolved", resolved, max_findings)
    return "\n".join(lines) + "\n"


def _describe_layout(layout: Layout) -> str:
    def names(labels: frozenset[str]) -> str:
        return ", ".join(sorted(labels)) or "none"

    return _escape(
        f"lexical labels {names(layout.lexical_labels)}; "
        f"community labels {names(layout.community_labels)}; "
        f"base labels {names(layout.base_labels)}; "
        f"names from {', '.join(layout.name_keys)}"
        + (f"; relationship types from {', '.join(layout.type_keys)}" if layout.type_keys else "")
        + (f"; expired facts by {', '.join(layout.expired_keys)}" if layout.expired_keys else "")
    )


def _describe_metadata(metadata: Mapping[str, str]) -> str:
    return _escape(", ".join(f"{key}={value}" for key, value in metadata.items()))


def _describe_trends(comparison: Comparison) -> str:
    counts = comparison.trend_counts
    description = ", ".join(f"{count:,} {trend}" for trend, count in counts.items())
    unscored = len(comparison.changes) - sum(counts.values())
    if unscored:
        description += f", {unscored:,} without two comparable scores"
    return description


def _caveats(base: Report, head: Report) -> list[str]:
    caveats = []
    if base.kgeval_version != head.kgeval_version:
        caveats.append(
            f"The reports come from kgeval {base.kgeval_version} and {head.kgeval_version}, "
            "whose metrics may differ."
        )
    if base.layout != head.layout:
        caveats.append("The reports read their graphs with different layouts.")
    return ["", *(_escape(caveat) for caveat in caveats)] if caveats else []


def _metadata_changes(base: Mapping[str, str], head: Mapping[str, str]) -> list[str]:
    keys = [key for key in dict.fromkeys([*head, *base]) if head.get(key) != base.get(key)]
    if not keys:
        return []
    lines = ["", "| Metadata | Base | Head |", "| --- | --- | --- |"]
    lines += [
        f"| {_escape(key)} | {_metadata_value(base, key)} | {_metadata_value(head, key)} |"
        for key in keys
    ]
    return lines


def _metadata_value(metadata: Mapping[str, str], key: str) -> str:
    return _escape(metadata[key]) if key in metadata else "—"


def _describe_run(report: Report) -> str:
    summary = report.summary
    return (
        f"kgeval {report.kgeval_version} on graph `{report.graph_fingerprint}`, with "
        f"{_count(summary.entities, 'entity', 'entities')} and {_count(summary.facts, 'fact')}"
    )


def _score_row(result: MetricResult) -> str:
    threshold = "—" if result.threshold is None else f"≥ {_percent(result.threshold)}"
    if result.score is None:
        outcome = f"skipped: {_escape(str(result.skip_reason))}"
        if result.passed is False:
            outcome = f"**FAIL**, {outcome}"
        return f"| {result.metric} | — | — | {threshold} | {outcome} |"
    score = _percent(result.score)
    if result.confidence_interval is not None:
        low, high = result.confidence_interval
        score += f" (95% CI {_percent(low)} to {_percent(high)})"
    outcome = {None: "", True: "pass", False: "**FAIL**"}[result.passed]
    failed = f"{result.failed:,} / {result.evaluated:,} {result.unit}"
    if result.unjudged:
        failed += f", {result.unjudged:,} unjudged"
    return f"| {result.metric} | {score} | {failed} | {threshold} | {outcome} |"


def _change_row(change: MetricChange) -> str:
    notes = []
    if change.base is None:
        notes.append("only in head")
    elif change.head is None:
        notes.append("only in base")
    elif not change.comparable:
        notes.append("parameters changed, so not compared")
    interval = change.change_interval
    if interval is not None:
        # State the trend of sampled scores, since only their interval tells noise from change.
        notes.append(str(change.trend))
    if change.regressed:
        notes.append("**REGRESSED**")
    new, resolved = f"{len(change.new_findings):,}", f"{len(change.resolved_findings):,}"
    if change.category is MetricCategory.JUDGED:
        new = resolved = "—"
        notes.append("findings of judged metrics are not compared")
    elif change.base is not None and change.head is not None and not change.comparable:
        new = resolved = "—"
    score_change = "—" if change.score_change is None else _points(change.score_change)
    if interval is not None:
        low, high = interval
        score_change += f" (95% CI {_points(low)} to {_points(high)})"
    return (
        f"| {change.metric} | {_score(change.base)} | {_score(change.head)} | {score_change} "
        f"| {new} | {resolved} | {'; '.join(notes)} |"
    )


def _score(result: MetricResult | None) -> str:
    if result is None:
        return "—"
    if result.score is None:
        return "skipped"
    # Show each score's unit count, since builds of different sizes score different populations.
    return f"{_percent(result.score)} of {result.evaluated:,}"


def _points(change: float) -> str:
    # Show changes in percentage points, which readers cannot mistake for relative changes.
    return f"{change * 100:+.1f} pp"


def _changed_findings(
    title: str, adjective: str, findings: Mapping[str, Sequence[Finding]], max_findings: int
) -> list[str]:
    lines = ["", f"## {title}"] if any(findings.values()) else []
    for metric, changed in findings.items():
        if changed:
            lines += ["", f"### {metric}: {_count(len(changed), f'{adjective} finding')}", ""]
            lines += _finding_items(changed, max_findings)
    return lines


def _finding_items(findings: Sequence[Finding], max_findings: int) -> list[str]:
    lines = [_finding_item(finding) for finding in findings[:max_findings]]
    hidden = len(findings) - max_findings
    if hidden > 0:
        lines.append(f"- and {_count(hidden, 'more finding')}")
    return lines


def _count(number: int, singular: str, plural: str | None = None) -> str:
    noun = singular if number == 1 else plural or f"{singular}s"
    return f"{number:,} {noun}"


def _finding_item(finding: Finding) -> str:
    lines = [f"- {_escape(finding.message)}"]
    for kind, ids in (("nodes", finding.node_ids), ("relationships", finding.relationship_ids)):
        if ids:
            shown = ", ".join(_code(element_id) for element_id in ids[:_MAX_IDS_SHOWN])
            hidden = len(ids) - _MAX_IDS_SHOWN
            lines.append(f"  - {kind}: {shown}" + (f" and {hidden:,} more" if hidden > 0 else ""))
    return "\n".join(lines)


def _code(text: str) -> str:
    # Fall back to escaped text when a backtick would end the code span early.
    return f"`{text}`" if "`" not in text else _escape(text)


def _percent(value: float) -> str:
    return f"{value:.1%}"


def _escape(text: str) -> str:
    return _MARKDOWN_SYNTAX.sub(lambda match: "\\" + match.group(), " ".join(text.split()))
