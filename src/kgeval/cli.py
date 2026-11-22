"""Provide the ``kgeval`` command-line interface."""

from __future__ import annotations

import inspect
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from neo4j import Driver, GraphDatabase
from neo4j.exceptions import DriverError
from pydantic import TypeAdapter, ValidationError
from rich.console import Console
from rich.markdown import Markdown

from kgeval._yaml import load_yaml
from kgeval.errors import InputError, KgevalError, SourceError
from kgeval.evaluation import evaluate
from kgeval.graph import GraphView, Layout
from kgeval.ground_truth import load_ground_truth
from kgeval.judges import PROVIDERS, CachedJudge, Judge, create_judge
from kgeval.metrics import (
    BUILTIN_METRICS,
    GROUND_TRUTH_METRICS,
    JUDGED_METRICS,
    RETRIEVAL_METRICS,
    SCHEMA_METRICS,
    Competency,
    Metric,
    MetricCategory,
    default_metrics,
)
from kgeval.questions import load_questions
from kgeval.report import Report, compare
from kgeval.retrieval import load_retrievals
from kgeval.schema import infer_schema, load_schema
from kgeval.sources import Neo4jSource

__all__ = ["app"]

app = typer.Typer(
    help="Evaluate knowledge graphs.",
    no_args_is_help=True,
    add_completion=False,
)


def _print_version(show: bool) -> None:
    if show:
        typer.echo(f"kgeval {version('kgeval')}")
        raise typer.Exit


@app.callback()
def _main(
    show_version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Print the installed kgeval version and exit.",
            is_eager=True,
            callback=_print_version,
        ),
    ] = False,
) -> None:
    pass


_ENABLED_BY = {
    **{metric_type.name: "--schema" for metric_type in SCHEMA_METRICS},
    **{metric_type.name: "--ground" for metric_type in GROUND_TRUTH_METRICS},
    Competency.name: "--questions",
    **{metric_type.name: "--judge" for metric_type in JUDGED_METRICS},
    **{metric_type.name: "--questions and --retrievals" for metric_type in RETRIEVAL_METRICS},
}


class OutputFormat(StrEnum):
    """Name the formats a report can be written in."""

    MARKDOWN = "markdown"
    JSON = "json"


Uri = Annotated[
    str | None,
    typer.Option(
        envvar="NEO4J_URI",
        help="Neo4j connection URI (default neo4j://localhost:7687); prompted for in a terminal.",
    ),
]
Username = Annotated[
    str | None,
    typer.Option(
        envvar="NEO4J_USERNAME", help="Neo4j user (default neo4j); prompted for in a terminal."
    ),
]
Password = Annotated[
    str | None,
    typer.Option(
        envvar="NEO4J_PASSWORD",
        help="Neo4j password; prompted for in a terminal; connects without auth if empty.",
    ),
]
Database = Annotated[
    str | None,
    typer.Option(envvar="NEO4J_DATABASE", help="Database to read; the server default if unset."),
]
LayoutFile = Annotated[
    Path | None,
    typer.Option(
        "--layout",
        help="YAML file of layout fields that replace the detected ones.",
        exists=True,
        dir_okay=False,
    ),
]
Output = Annotated[
    Path | None, typer.Option("--output", "-o", help="Write to this file, not standard output.")
]
MaxFindings = Annotated[int, typer.Option(min=0, help="Findings listed per metric in Markdown.")]
ReportFile = Annotated[
    Path,
    typer.Argument(
        help="A report that `kgeval run --format json` wrote.", exists=True, dir_okay=False
    ),
]
Verbose = Annotated[bool, typer.Option("--verbose", "-v", help="Log progress to standard error.")]


@app.command()
def run(
    uri: Uri = None,
    username: Username = None,
    password: Password = None,
    database: Database = None,
    schema: Annotated[
        Path | None,
        typer.Option(
            help="Graph schema in neo4j-graphrag format (YAML or JSON); adds the schema metrics.",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    ground_truth: Annotated[
        Path | None,
        typer.Option(
            "--ground",
            help="Ground truth facts (YAML or JSON); adds precision, recall, and F1.",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    questions: Annotated[
        Path | None,
        typer.Option(
            help="Test questions with their supporting facts (YAML or JSON); adds competency.",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    retrievals: Annotated[
        Path | None,
        typer.Option(
            help="Element IDs a retriever returned per question (YAML or JSON); "
            "adds the retrieval metrics. Needs --questions.",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    judge_model: Annotated[
        str | None,
        typer.Option(
            "--judge",
            help="Model to judge with: an OpenAI model, or PROVIDER:MODEL for "
            f"{', '.join(provider for provider in PROVIDERS if provider != 'openai')}; "
            "adds the judged metrics. Needs the provider's API key.",
        ),
    ] = None,
    judge_base_url: Annotated[
        str | None,
        typer.Option(help="Base URL that replaces the provider's, such as another server's."),
    ] = None,
    judge_cache: Annotated[
        Path | None,
        typer.Option(
            help="SQLite file that stores verdicts, so reruns reuse them.", dir_okay=False
        ),
    ] = None,
    sample_size: Annotated[
        int, typer.Option(min=1, help="Most units each sampled judged metric judges.")
    ] = 100,
    seed: Annotated[int, typer.Option(help="Seed that selects which units are sampled.")] = 0,
    top_k: Annotated[
        int | None,
        typer.Option(
            min=1, help="Top-ranked element IDs of each retrieval to evaluate; all if unset."
        ),
    ] = None,
    metric: Annotated[
        list[str] | None,
        typer.Option("--metric", "-m", help="Run only this metric; repeat for several."),
    ] = None,
    threshold: Annotated[
        list[str] | None,
        typer.Option(
            "--threshold",
            "-t",
            help="Minimum score as METRIC=VALUE, VALUE from 0 to 1; repeat for several.",
        ),
    ] = None,
    metadata: Annotated[
        list[str] | None,
        typer.Option(
            help="Run description as KEY=VALUE, such as model=gpt-5-mini; repeat for several.",
        ),
    ] = None,
    layout: LayoutFile = None,
    output_format: Annotated[
        OutputFormat, typer.Option("--format", "-f", help="Report format.")
    ] = OutputFormat.MARKDOWN,
    output: Output = None,
    max_findings: MaxFindings = 10,
    verbose: Verbose = False,
) -> None:
    """Evaluate a Neo4j knowledge graph and report scores and findings.

    Exits with status 1 when a metric scores below its threshold and 2 on errors.
    """
    _configure_logging(verbose)
    thresholds = _parse_thresholds(threshold or [])
    run_metadata = _parse_metadata(metadata or [])
    for option, value in (("--judge-base-url", judge_base_url), ("--judge-cache", judge_cache)):
        if value is not None and judge_model is None:
            raise typer.BadParameter("needs --judge", param_hint=option)
    if retrievals is not None and questions is None:
        raise typer.BadParameter("needs --questions", param_hint="--retrievals")
    if top_k is not None and retrievals is None:
        raise typer.BadParameter("needs --retrievals", param_hint="--top-k")
    try:
        layout_overrides = _load_layout(layout)
        judge: Judge | None = None
        if judge_model is not None:
            judge = create_judge(judge_model, base_url=judge_base_url)
            if judge_cache is not None:
                judge = CachedJudge(judge, judge_cache)
        metrics = default_metrics(
            schema=None if schema is None else load_schema(schema),
            ground_truth=None if ground_truth is None else load_ground_truth(ground_truth),
            questions=None if questions is None else load_questions(questions),
            retrievals=None if retrievals is None else load_retrievals(retrievals),
            judge=judge,
            sample_size=sample_size,
            seed=seed,
            top_k=top_k,
        )
        metrics = _select(metrics, metric or [], thresholds)
        with _connect(uri, username, password) as driver:
            graph = Neo4jSource(driver, database).load()
        report = evaluate(
            graph,
            metrics,
            layout=Layout.detect(graph, layout_overrides),
            metadata=run_metadata,
        )
    except KgevalError as error:
        _fail(str(error))
    if output_format is OutputFormat.JSON:
        _write(report.model_dump_json(indent=2) + "\n", output)
    else:
        _write(report.to_markdown(max_findings), output, render_markdown=True)
    if not report.passed:
        raise typer.Exit(1)


@app.command("compare")
def compare_command(
    base: ReportFile,
    head: ReportFile,
    max_drop: Annotated[
        float,
        typer.Option(
            min=0, max=1, help="Largest score drop, from 0 to 1, that is not a regression."
        ),
    ] = 0.0,
    output: Output = None,
    max_findings: MaxFindings = 10,
) -> None:
    """Compare the report HEAD with the report BASE, such as a baseline, metric by metric.

    Exits with status 1 when a metric regressed, because HEAD scores it lower than BASE
    by more than --max-drop and sampling explains, or no longer scores a metric that
    BASE scored, and 2 on errors.
    """
    try:
        comparison = compare(_load_report(base), _load_report(head), max_drop=max_drop)
    except KgevalError as error:
        _fail(str(error))
    _write(comparison.to_markdown(max_findings), output, render_markdown=True)
    if comparison.regressed:
        raise typer.Exit(1)


@app.command("infer-schema")
def infer_schema_command(
    uri: Uri = None,
    username: Username = None,
    password: Password = None,
    database: Database = None,
    layout: LayoutFile = None,
    output: Output = None,
    verbose: Verbose = False,
) -> None:
    """Write the schema the graph follows today, as YAML to review and pass to --schema."""
    _configure_logging(verbose)
    try:
        layout_overrides = _load_layout(layout)
        with _connect(uri, username, password) as driver:
            graph = Neo4jSource(driver, database).load()
        schema = infer_schema(GraphView(graph, Layout.detect(graph, layout_overrides)))
    except KgevalError as error:
        _fail(str(error))
    _write(schema.to_yaml(), output)


@app.command("metrics")
def metrics_command() -> None:
    """List every built-in metric by category, with what each category needs."""
    lines = ["# kgeval metrics"]
    for category in MetricCategory:
        metric_types = [
            metric_type for metric_type in BUILTIN_METRICS if metric_type.category is category
        ]
        options = " or ".join(
            dict.fromkeys(
                _ENABLED_BY[metric_type.name]
                for metric_type in metric_types
                if metric_type.name in _ENABLED_BY
            )
        )
        lines += [
            "",
            f"## {category.value} ({f'needs {options}' if options else 'runs by default'})",
            "",
            "| Metric | Description |",
            "| --- | --- |",
        ]
        lines += [
            f"| {metric_type.name} | {_summary(metric_type)} |" for metric_type in metric_types
        ]
    _write("\n".join(lines) + "\n", None, render_markdown=True)


def _summary(metric_type: type[Metric]) -> str:
    return (inspect.getdoc(metric_type) or "").splitlines()[0]


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _ask(value: str | None, text: str, default: str, *, hide_input: bool = False) -> str:
    if value is not None:
        return value
    # Prompt only in a terminal, so pipelines fall back to the default instead of blocking.
    if not sys.stdin.isatty():
        return default
    # Write prompts to stderr, so they stay visible when the report is redirected.
    return str(
        typer.prompt(
            text, default=default, hide_input=hide_input, show_default=not hide_input, err=True
        )
    )


@contextmanager
def _connect(uri: str | None, username: str | None, password: str | None) -> Iterator[Driver]:
    uri = _ask(uri, "Neo4j URI", "neo4j://localhost:7687")
    username = _ask(username, "Neo4j username", "neo4j")
    password = _ask(password, "Neo4j password (empty for none)", "", hide_input=True)
    auth = (username, password) if password else None
    try:
        driver = GraphDatabase.driver(uri, auth=auth)
    except (ValueError, DriverError) as error:
        raise SourceError(f"cannot connect to {uri}: {error}") from error
    with driver:
        yield driver


def _parse_thresholds(values: list[str]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for value in values:
        name, separator, number = value.partition("=")
        try:
            score = float(number)
        except ValueError:
            score = float("nan")
        # Rely on NaN failing every comparison, so "nan" and unparsable values are rejected.
        if not (separator and name.strip() and 0 <= score <= 1):
            raise typer.BadParameter(
                f"expected METRIC=VALUE with VALUE from 0 to 1, got {value!r}",
                param_hint="--threshold",
            )
        thresholds[name.strip()] = score
    return thresholds


def _parse_metadata(values: list[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for value in values:
        key, separator, text = value.partition("=")
        key, text = key.strip(), text.strip()
        if not (separator and key and text):
            raise typer.BadParameter(f"expected KEY=VALUE, got {value!r}", param_hint="--metadata")
        if key in metadata:
            raise typer.BadParameter(f"repeats the key {key!r}", param_hint="--metadata")
        metadata[key] = text
    return metadata


def _select(metrics: list[Metric], names: list[str], thresholds: dict[str, float]) -> list[Metric]:
    enabled = [metric.name for metric in metrics]
    missing = sorted((set(names) | set(thresholds)) - set(enabled))
    problems = [f"{name} needs {_ENABLED_BY[name]}" for name in missing if name in _ENABLED_BY]
    unknown = [name for name in missing if name not in _ENABLED_BY]
    if unknown:
        noun = "metric" if len(unknown) == 1 else "metrics"
        problems.append(f"unknown {noun} {', '.join(unknown)}; choose from {', '.join(enabled)}")
    unselected = sorted(set(thresholds) - set(names)) if names else []
    if unselected:
        problems.append(f"--threshold names {', '.join(unselected)}, which --metric excludes")
    if problems:
        raise typer.BadParameter("; ".join(problems))
    return [
        # Rely on _parse_thresholds for the range check, because model_copy does not validate.
        metric.model_copy(update={"threshold": thresholds[metric.name]})
        if metric.name in thresholds
        else metric
        for metric in metrics
        if not names or metric.name in names
    ]


def _load_report(path: Path) -> Report:
    try:
        return Report.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise InputError(f"invalid report {path}: {error}") from error


def _load_layout(path: Path | None) -> Layout | None:
    return None if path is None else load_yaml(path, TypeAdapter(Layout))


def _write(text: str, path: Path | None, *, render_markdown: bool = False) -> None:
    if path is not None:
        try:
            path.write_text(text, encoding="utf-8")
        except OSError as error:
            _fail(f"cannot write {path}: {error}")
    elif render_markdown and sys.stdout.isatty():
        Console().print(Markdown(text))
    else:
        # Write UTF-8 bytes, since graph text can hold characters the console code page lacks.
        sys.stdout.flush()
        sys.stdout.buffer.write(text.encode("utf-8"))
        sys.stdout.buffer.flush()


def _fail(message: str) -> NoReturn:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(2)
