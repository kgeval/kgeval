"""Implement the ``kgeval`` command-line interface."""

from __future__ import annotations

import inspect
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
from neo4j import Driver, GraphDatabase
from neo4j.exceptions import DriverError
from pydantic import TypeAdapter
from rich.console import Console
from rich.markdown import Markdown

from kgeval._files import load_json, load_yaml
from kgeval.errors import KgevalError, SourceError
from kgeval.evaluation import evaluate
from kgeval.graph import Graph, GraphView, Layout, load_graph
from kgeval.ground_truth import load_ground_truth
from kgeval.metrics import (
    BUILTIN_METRICS,
    InputName,
    Inputs,
    Metric,
    MetricCategory,
    load_suite,
)
from kgeval.metrics.model import DEFAULT_SAMPLE_SIZE
from kgeval.model import Model
from kgeval.questions import load_questions
from kgeval.report import DEFAULT_MAX_FINDINGS, Report, compare
from kgeval.retrieval import load_retrievals
from kgeval.schema import infer_schema, load_schema
from kgeval.sources import Neo4jSource

__all__ = ["app"]

app = typer.Typer(
    help="An evaluation framework for knowledge graphs.",
    no_args_is_help=True,
    add_completion=False,
)

_GRAPH = "Graph"
_INPUTS = "Inputs"
_MODEL = "Model"
_METRICS = "Metrics"
_OUTPUT = "Output"
_REPORT = "A report that `kgeval run --format json` or `pytest --kgeval-report` wrote."

_OPTIONS: dict[InputName, str] = {
    "schema": "--schema",
    "ground_truth": "--ground-truth",
    "questions": "--questions",
    "retrievals": "--retrievals",
    "model": "--model",
}


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


class OutputFormat(StrEnum):
    """Name the formats a report can be written in."""

    MARKDOWN = "markdown"
    JSON = "json"


_Uri = Annotated[
    str | None,
    typer.Option(
        envvar="NEO4J_URI",
        help="Neo4j connection URI (default neo4j://localhost:7687); prompted for in a terminal.",
        rich_help_panel=_GRAPH,
    ),
]
_Username = Annotated[
    str | None,
    typer.Option(
        envvar="NEO4J_USERNAME",
        help="Neo4j user (default neo4j); prompted for in a terminal.",
        rich_help_panel=_GRAPH,
    ),
]
_Password = Annotated[
    str | None,
    typer.Option(
        envvar="NEO4J_PASSWORD",
        help="Neo4j password; prompted for in a terminal; connects without auth if empty.",
        rich_help_panel=_GRAPH,
    ),
]
_Database = Annotated[
    str | None,
    typer.Option(
        envvar="NEO4J_DATABASE",
        help="Database to read; the server default if unset.",
        rich_help_panel=_GRAPH,
    ),
]
_GraphFile = Annotated[
    Path | None,
    typer.Option(
        "--graph",
        help="Graph snapshot (JSON) to read instead of Neo4j, ignoring the connection options.",
        exists=True,
        dir_okay=False,
        rich_help_panel=_GRAPH,
    ),
]
_LayoutFile = Annotated[
    Path | None,
    typer.Option(
        "--layout",
        help="YAML file of layout fields that replace the detected ones.",
        exists=True,
        dir_okay=False,
        rich_help_panel=_GRAPH,
    ),
]
_Output = Annotated[
    Path | None,
    typer.Option(
        "--output", "-o", help="Write to this file, not standard output.", rich_help_panel=_OUTPUT
    ),
]
_MaxFindings = Annotated[
    int,
    typer.Option(min=0, help="Findings listed per metric in Markdown.", rich_help_panel=_OUTPUT),
]
_Verbose = Annotated[
    bool,
    typer.Option(
        "--verbose", "-v", help="Log progress to standard error.", rich_help_panel=_OUTPUT
    ),
]


@app.command()
def run(
    uri: _Uri = None,
    username: _Username = None,
    password: _Password = None,
    database: _Database = None,
    graph_file: _GraphFile = None,
    layout: _LayoutFile = None,
    schema: Annotated[
        Path | None,
        typer.Option(
            help="Graph schema in neo4j-graphrag format (YAML or JSON).",
            exists=True,
            dir_okay=False,
            rich_help_panel=_INPUTS,
        ),
    ] = None,
    ground_truth: Annotated[
        Path | None,
        typer.Option(
            help="Facts the graph should contain (YAML or JSON).",
            exists=True,
            dir_okay=False,
            rich_help_panel=_INPUTS,
        ),
    ] = None,
    questions: Annotated[
        Path | None,
        typer.Option(
            help="Test questions with their supporting facts (YAML or JSON).",
            exists=True,
            dir_okay=False,
            rich_help_panel=_INPUTS,
        ),
    ] = None,
    retrievals: Annotated[
        Path | None,
        typer.Option(
            help="Element IDs a retriever returned per question (YAML or JSON).",
            exists=True,
            dir_okay=False,
            rich_help_panel=_INPUTS,
        ),
    ] = None,
    model_name: Annotated[
        str | None,
        typer.Option(
            "--model",
            help="Model as LiteLLM names it, such as openai/gpt-5-mini; needs the provider's "
            "API key.",
            rich_help_panel=_MODEL,
        ),
    ] = None,
    model_base_url: Annotated[
        str | None,
        typer.Option(
            help="Base URL that replaces the provider's, such as another server's.",
            rich_help_panel=_MODEL,
        ),
    ] = None,
    model_cache: Annotated[
        Path | None,
        typer.Option(
            help="SQLite file that stores verdicts, so reruns reuse them.",
            dir_okay=False,
            rich_help_panel=_MODEL,
        ),
    ] = None,
    metric: Annotated[
        list[str] | None,
        typer.Option(
            "--metric",
            "-m",
            help="Run only this metric; repeat for several.",
            rich_help_panel=_METRICS,
        ),
    ] = None,
    suite: Annotated[
        Path | None,
        typer.Option(
            help="Metrics to run, with their settings (YAML or JSON).",
            exists=True,
            dir_okay=False,
            rich_help_panel=_METRICS,
        ),
    ] = None,
    threshold: Annotated[
        list[str] | None,
        typer.Option(
            "--threshold",
            "-t",
            help="Minimum score as METRIC=VALUE, VALUE from 0 to 1; repeat for several.",
            rich_help_panel=_METRICS,
        ),
    ] = None,
    sample_size: Annotated[
        int,
        typer.Option(
            min=1,
            help="Most units each sampled metric checks, unless the suite sets it.",
            rich_help_panel=_METRICS,
        ),
    ] = DEFAULT_SAMPLE_SIZE,
    seed: Annotated[
        int,
        typer.Option(
            help="Seed that selects which units are sampled, unless the suite sets it.",
            rich_help_panel=_METRICS,
        ),
    ] = 0,
    top_k: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Top-ranked element IDs of each retrieval to evaluate, unless the suite sets "
            "k; all if unset.",
            rich_help_panel=_METRICS,
        ),
    ] = None,
    output_format: Annotated[
        OutputFormat,
        typer.Option("--format", "-f", help="Report format.", rich_help_panel=_OUTPUT),
    ] = OutputFormat.MARKDOWN,
    output: _Output = None,
    max_findings: _MaxFindings = DEFAULT_MAX_FINDINGS,
    metadata: Annotated[
        list[str] | None,
        typer.Option(
            help="Run description as KEY=VALUE, such as chunk_size=512; repeat for several.",
            rich_help_panel=_OUTPUT,
        ),
    ] = None,
    verbose: _Verbose = False,
) -> None:
    """Evaluate a knowledge graph in Neo4j, or a snapshot of one, and report scores and findings.

    Without --metric or --suite, every metric whose inputs are given runs. Exits with
    status 1 when a metric scores below its threshold and 2 on errors.
    """
    _configure_logging(verbose)
    thresholds = _parse_thresholds(threshold or [])
    run_metadata = _parse_metadata(metadata or [])
    for option, value in (("--model-base-url", model_base_url), ("--model-cache", model_cache)):
        if value is not None and model_name is None:
            raise typer.BadParameter("needs --model", param_hint=option)
    if retrievals is not None and questions is None:
        raise typer.BadParameter("needs --questions", param_hint="--retrievals")
    try:
        layout_overrides = _load_layout(layout)
        given: dict[str, Any] = {
            "schema": None if schema is None else load_schema(schema),
            "ground_truth": None if ground_truth is None else load_ground_truth(ground_truth),
            "questions": None if questions is None else load_questions(questions),
            "retrievals": None if retrievals is None else load_retrievals(retrievals),
            "model": None
            if model_name is None
            else Model(model_name, base_url=model_base_url, cache=model_cache),
        }
        metrics = _select(
            load_suite(suite) if suite else [metric_type() for metric_type in BUILTIN_METRICS],
            metric or [],
            thresholds,
            Inputs(**given),
            suite,
            {"sample_size": sample_size, "seed": seed, "k": top_k},
        )
        graph = _load_graph(graph_file, uri, username, password, database)
        report = evaluate(
            graph,
            metrics,
            layout=Layout.detect(graph, layout_overrides),
            metadata=run_metadata,
            **given,
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
    # Name the arguments, since typer's default names differ between its versions.
    base: Annotated[
        Path, typer.Argument(metavar="BASE", help=_REPORT, exists=True, dir_okay=False)
    ],
    head: Annotated[
        Path, typer.Argument(metavar="HEAD", help=_REPORT, exists=True, dir_okay=False)
    ],
    max_drop: Annotated[
        float,
        typer.Option(
            min=0, max=1, help="Largest score drop, from 0 to 1, that is not a regression."
        ),
    ] = 0.0,
    output: _Output = None,
    max_findings: _MaxFindings = DEFAULT_MAX_FINDINGS,
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
    uri: _Uri = None,
    username: _Username = None,
    password: _Password = None,
    database: _Database = None,
    graph_file: _GraphFile = None,
    layout: _LayoutFile = None,
    output: _Output = None,
    verbose: _Verbose = False,
) -> None:
    """Write the schema the graph follows today, as YAML to review and pass to --schema."""
    _configure_logging(verbose)
    try:
        layout_overrides = _load_layout(layout)
        graph = _load_graph(graph_file, uri, username, password, database)
        schema = infer_schema(GraphView(graph, Layout.detect(graph, layout_overrides)))
    except KgevalError as error:
        _fail(str(error))
    _write(schema.to_yaml(), output)


@app.command()
def snapshot(
    uri: _Uri = None,
    username: _Username = None,
    password: _Password = None,
    database: _Database = None,
    output: _Output = None,
    verbose: _Verbose = False,
) -> None:
    """Write a snapshot of the graph in Neo4j as JSON, for --graph to read."""
    _configure_logging(verbose)
    try:
        graph = _load_graph(None, uri, username, password, database)
    except KgevalError as error:
        _fail(str(error))
    _write(graph.model_dump_json(indent=2) + "\n", output)


@app.command("metrics")
def metrics_command() -> None:
    """List every built-in metric by category, with the inputs it needs."""
    lines = ["# kgeval metrics"]
    for category in MetricCategory:
        lines += ["", f"## {category.value}", "", "| Metric | Needs | Description |"]
        lines.append("| --- | --- | --- |")
        lines += [
            f"| {metric_type.name} | {_needs(metric_type.needs)} | {_summary(metric_type)} |"
            for metric_type in BUILTIN_METRICS
            if metric_type.category is category
        ]
    _write("\n".join(lines) + "\n", None, render_markdown=True)


def _needs(needs: tuple[InputName, ...]) -> str:
    return " and ".join(_OPTIONS[need] for need in needs) or "the graph only"


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
    # Prompt only when stdin and stderr are terminals, since Windows calls NUL a terminal too.
    if not (sys.stdin.isatty() and sys.stderr.isatty()):
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


def _load_graph(
    path: Path | None,
    uri: str | None,
    username: str | None,
    password: str | None,
    database: str | None,
) -> Graph:
    if path is not None:
        return load_graph(path)
    with _connect(uri, username, password) as driver:
        return Neo4jSource(driver, database).load()


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


def _select(
    candidates: list[Metric],
    names: list[str],
    thresholds: dict[str, float],
    inputs: Inputs,
    suite: Path | None,
    settings: dict[str, int | None],
) -> list[Metric]:
    available = [metric.name for metric in candidates]
    problems = []
    for option, given in (("--metric", names), ("--threshold", list(thresholds))):
        if unknown := sorted(set(given) - set(available)):
            listed = ", ".join(unknown)
            if suite is not None:
                problems.append(f"{option} names {listed}, which {suite} does not list")
            else:
                noun = "metric" if len(unknown) == 1 else "metrics"
                problems.append(f"{option} names unknown {noun} {listed}")
    if problems:
        problems[-1] += f"; choose from {', '.join(available)}"
    if names and (unselected := sorted((set(thresholds) & set(available)) - set(names))):
        problems.append(f"--threshold names {', '.join(unselected)}, which --metric excludes")
    selected = [metric for metric in candidates if not names or metric.name in names]
    if not names and suite is None:
        # Keep a metric that --threshold names, so that its missing inputs are reported.
        selected = [
            metric
            for metric in selected
            if not inputs.missing(metric.needs) or metric.name in thresholds
        ]
    problems += [
        f"{metric.name} needs {_needs(tuple(missing))}"
        for metric in selected
        if (missing := inputs.missing(metric.needs))
    ]
    if problems:
        raise typer.BadParameter("; ".join(problems))
    configured = []
    for metric in selected:
        # Keep the suite's settings, and rely on option checks, since model_copy does not validate.
        update: dict[str, Any] = {
            field: value
            for field, value in settings.items()
            if value is not None
            and field in type(metric).model_fields
            and field not in metric.model_fields_set
        }
        if metric.name in thresholds:
            update["threshold"] = thresholds[metric.name]
        configured.append(metric.model_copy(update=update))
    return configured


def _load_report(path: Path) -> Report:
    return load_json(path, TypeAdapter(Report))


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
