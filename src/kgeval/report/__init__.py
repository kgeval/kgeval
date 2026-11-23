"""Provide reports and report comparisons."""

from kgeval.report.comparison import Comparison, MetricChange, Trend, compare
from kgeval.report.markdown import DEFAULT_MAX_FINDINGS
from kgeval.report.model import REPORT_FORMAT, GraphSummary, Report

__all__ = [
    "DEFAULT_MAX_FINDINGS",
    "REPORT_FORMAT",
    "Comparison",
    "GraphSummary",
    "MetricChange",
    "Report",
    "Trend",
    "compare",
]
