"""Test the ``kgeval`` command-line interface."""

import io
import json
import re
import sys
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace, TracebackType

import pytest
import typer
from typer.testing import CliRunner

from kgeval import cli
from kgeval.errors import SourceError
from kgeval.graph.model import Graph
from kgeval.metrics.catalog import BUILTIN_METRICS
from kgeval.schema.model import load_schema
from tests.doubles import ScriptedModel
from tests.factories import EXAMPLE_DIR, EXAMPLE_RETRIEVALS, example_graph


class FakeDriver:
    def __enter__(self) -> "FakeDriver":
        return self

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


class FakeSource:
    graph: Graph | None = example_graph()

    def __init__(self, driver: FakeDriver, database: str | None = None) -> None:
        self.database = database

    def load(self) -> Graph:
        if self.graph is None:
            raise SourceError("cannot read the graph from Neo4j: connection refused")
        return self.graph


def plain(output: str) -> str:
    """Return CLI output without the colors, box drawing, and line wrapping of error panels."""
    output = re.sub(r"\x1b\[[0-9;]*m", "", output)
    return " ".join(output.translate(str.maketrans("│╭╮╰╯─┌┐└┘", "          ")).split())


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> CliRunner:
    monkeypatch.setattr(
        cli, "GraphDatabase", SimpleNamespace(driver=lambda uri, auth: FakeDriver())
    )
    monkeypatch.setattr(cli, "Neo4jSource", FakeSource)
    return CliRunner()


def test_run_prints_a_markdown_report(runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["run"])

    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("# kgeval report\n")
    assert "| entity_uniqueness | 90.0% |" in result.stdout


def test_connect_prompts_for_unset_settings_only_in_a_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connected: list[tuple[str, object]] = []

    def driver(uri: str, auth: object) -> FakeDriver:
        connected.append((uri, auth))
        return FakeDriver()

    monkeypatch.setattr(cli, "GraphDatabase", SimpleNamespace(driver=driver))
    typed = iter(["neo4j+s://db.example", "admin", "secret"])
    monkeypatch.setattr(typer, "prompt", lambda *args, **kwargs: next(typed))

    for stdin, stderr in ((True, True), (True, False), (False, True)):
        monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda tty=stdin: tty))
        monkeypatch.setattr(sys.stderr, "isatty", lambda tty=stderr: tty)
        with cli._connect(None, None, None):
            pass

    assert connected == [
        ("neo4j+s://db.example", ("admin", "secret")),
        ("neo4j://localhost:7687", None),
        ("neo4j://localhost:7687", None),
    ]


def test_run_exits_with_1_when_a_threshold_fails(runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["run", "-t", "entity_uniqueness=0.9", "-t", "entity_naming=1"])

    assert result.exit_code == 1
    assert "**FAILED**" in result.stdout


def test_run_writes_json_reports(runner: CliRunner, tmp_path: Path) -> None:
    path = tmp_path / "report.json"

    result = runner.invoke(cli.app, ["run", "--format", "json", "--output", str(path)])

    assert result.exit_code == 0, result.output
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["results"][0]["metric"] == "entity_uniqueness"
    assert report["results"][0]["score"] == pytest.approx(9 / 10)


def test_run_records_metadata_in_the_report(runner: CliRunner) -> None:
    result = runner.invoke(
        cli.app,
        ["run", "--format", "json", "--metadata", "model=gpt-5-mini", "--metadata", "prompt = v2"],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["metadata"] == {"model": "gpt-5-mini", "prompt": "v2"}


def test_run_layout_file_replaces_only_the_fields_it_sets(
    runner: CliRunner, tmp_path: Path
) -> None:
    layout = tmp_path / "layout.yaml"
    layout.write_text("name_keys: [fileName]\n", encoding="utf-8")

    result = runner.invoke(cli.app, ["run", "--layout", str(layout), "--format", "json"])

    assert result.exit_code == 0, result.output
    report_layout = json.loads(result.stdout)["layout"]
    assert report_layout["name_keys"] == ["fileName"]
    assert report_layout["base_labels"] == ["__Entity__"]
    assert report_layout["lexical_labels"] == ["Chunk", "Document"]


def test_run_evaluates_the_chosen_metrics_with_the_example_inputs(
    runner: CliRunner, tmp_path: Path
) -> None:
    retrievals = tmp_path / "retrievals.json"
    retrievals.write_text(json.dumps(EXAMPLE_RETRIEVALS), encoding="utf-8")

    result = runner.invoke(
        cli.app,
        [
            "run",
            *("--ground-truth", str(EXAMPLE_DIR / "ground_truth.yaml")),
            *("--schema", str(EXAMPLE_DIR / "schema.yaml")),
            *("--questions", str(EXAMPLE_DIR / "questions.yaml")),
            *("--retrievals", str(retrievals)),
            *("--format", "json"),
            *("-m", "entity_conformance", "-m", "fact_precision"),
            *("-m", "competency", "-m", "retrieval_recall"),
        ],
    )

    assert result.exit_code == 0, result.output
    counts = {
        item["metric"]: (item["failed"], item["evaluated"])
        for item in json.loads(result.stdout)["results"]
    }
    assert counts == {
        "entity_conformance": (3, 11),
        "fact_precision": (2, 5),
        "competency": (1, 5),
        "retrieval_recall": (2, 6),
    }


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["-t", "entity_uniqueness"], "METRIC=VALUE"),
        (["-t", "entity_uniqueness=1.5"], "METRIC=VALUE"),
        (["-t", "entity_uniqueness=nan"], "METRIC=VALUE"),
        (["-m", "fact_accuracy"], "fact_accuracy needs --model"),
        (["-m", "retrieval_recall"], "retrieval_recall needs --questions and --retrievals"),
        (["--retrievals", str(EXAMPLE_DIR / "questions.yaml")], "needs --questions"),
        (["--model-base-url", "http://localhost:8000/v1"], "needs --model"),
        (["--model-cache", "verdicts.sqlite"], "needs --model"),
        (["-t", "fact_precision=0.9"], "fact_precision needs --ground-truth"),
        (["-m", "entity_unicorns"], "unknown metric entity_unicorns; choose from"),
        (["--metadata", "model"], "expected KEY=VALUE, got 'model'"),
        (["--metadata", "=gpt-5"], "expected KEY=VALUE, got '=gpt-5'"),
        (["--metadata", "model=a", "--metadata", "model=b"], "repeats the key 'model'"),
        (
            ["-m", "entity_naming", "-t", "entity_uniqueness=0.9"],
            "--threshold names entity_uniqueness, which --metric excludes",
        ),
    ],
)
def test_run_rejects_bad_options(runner: CliRunner, arguments: list[str], message: str) -> None:
    result = runner.invoke(cli.app, ["run", *arguments])

    assert result.exit_code == 2
    assert message in plain(result.output)


def test_run_reports_source_errors(runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(FakeSource, "graph", None)

    result = runner.invoke(cli.app, ["run"])

    assert result.exit_code == 2
    assert result.stderr == "Error: cannot read the graph from Neo4j: connection refused\n"


def test_reports_reach_redirected_output_as_utf8(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="cp1252"))

    cli._write("≥ 99% for 新加坡\n", None)

    assert raw.getvalue().decode("utf-8") == "≥ 99% for 新加坡\n"


def test_run_reports_unwritable_output(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["run", "--output", str(tmp_path / "missing" / "report.md")])

    assert result.exit_code == 2
    assert result.stderr.startswith("Error: cannot write ")


def test_run_creates_the_model_from_its_options(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[tuple[str, str | None, Path | None]] = []

    def create(
        model: str, *, base_url: str | None = None, cache: Path | None = None
    ) -> ScriptedModel:
        created.append((model, base_url, cache))
        return ScriptedModel()

    monkeypatch.setattr(cli, "Model", create)

    result = runner.invoke(
        cli.app,
        ["run", "--model", "groq/test-model", "-m", "fact_accuracy", "-m", "node_type_accuracy"],
    )

    assert result.exit_code == 0, result.output
    assert created == [("groq/test-model", None, None)]
    assert "| fact_accuracy | 100.0% |" in result.stdout
    assert "| node_type_accuracy | 100.0% |" in result.stdout


def test_run_evaluates_the_metrics_a_suite_lists_with_their_settings(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    suite = tmp_path / "suite.yaml"
    suite.write_text("entity_naming:\nfact_accuracy: {threshold: 0.9}\n", encoding="utf-8")
    monkeypatch.setattr(cli, "Model", lambda name, **settings: ScriptedModel())
    run = ["run", "--model", "groq/test-model", "--suite", str(suite), "--format", "json"]

    def results(*arguments: str) -> list[tuple[str, float | None]]:
        result = runner.invoke(cli.app, [*run, *arguments])
        assert result.exit_code == 0, result.output
        return [
            (item["metric"], item["threshold"]) for item in json.loads(result.stdout)["results"]
        ]

    assert results() == [("entity_naming", None), ("fact_accuracy", 0.9)]
    assert results("-m", "fact_accuracy", "-t", "fact_accuracy=0.5") == [("fact_accuracy", 0.5)]
    result = runner.invoke(cli.app, [*run, "-m", "entity_uniqueness"])
    assert result.exit_code == 2
    assert "does not list entity_uniqueness; choose from entity_naming, fact_accuracy" in plain(
        result.output
    )


def test_run_names_a_missing_model_key(runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    result = runner.invoke(cli.app, ["run", "--model", "groq/test-model"])

    assert result.exit_code == 2
    assert result.stderr == "Error: missing environment variables: GROQ_API_KEY\n"


def test_version_prints_the_installed_version(runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["--version"])

    assert result.exit_code == 0
    assert result.output == f"kgeval {version('kgeval')}\n"


def test_metrics_lists_every_builtin_metric_with_the_inputs_it_needs(runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["metrics"])

    assert result.exit_code == 0, result.output
    assert "| entity_uniqueness | the graph only |" in result.stdout
    assert "| fact_precision | --ground-truth |" in result.stdout
    assert "| fact_accuracy | --model |" in result.stdout
    assert "| retrieval_recall | --questions and --retrievals |" in result.stdout
    assert all(f"| {metric_type.name} |" in result.stdout for metric_type in BUILTIN_METRICS)


def test_infer_schema_writes_a_schema_that_loads_back(runner: CliRunner, tmp_path: Path) -> None:
    path = tmp_path / "schema.yaml"

    result = runner.invoke(cli.app, ["infer-schema", "--output", str(path)])

    assert result.exit_code == 0, result.output
    schema = load_schema(path)
    assert [node_type.label for node_type in schema.node_types][:2] == ["Person", "Organization"]


def test_snapshot_writes_a_graph_that_run_and_infer_schema_read_without_neo4j(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "graph.json"
    assert runner.invoke(cli.app, ["snapshot", "--output", str(path)]).exit_code == 0
    # Fail every read from Neo4j, so only the snapshot can supply the graph.
    monkeypatch.setattr(FakeSource, "graph", None)

    report = runner.invoke(cli.app, ["run", "--graph", str(path)])
    schema = runner.invoke(cli.app, ["infer-schema", "--graph", str(path)])

    assert report.exit_code == 0, report.output
    assert "| entity_uniqueness | 90.0% |" in report.stdout
    assert schema.exit_code == 0, schema.output
    assert "label: Person" in schema.stdout


def run_json(runner: CliRunner, path: Path) -> Path:
    result = runner.invoke(cli.app, ["run", "--format", "json", "--output", str(path)])
    assert result.exit_code == 0, result.output
    return path


def test_compare_exits_with_0_when_nothing_regressed(runner: CliRunner, tmp_path: Path) -> None:
    report = run_json(runner, tmp_path / "report.json")

    result = runner.invoke(cli.app, ["compare", str(report), str(report)])

    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("# kgeval comparison\n\n**NO REGRESSION**\n")


def test_compare_exits_with_1_when_the_head_scores_worse_with_a_new_defect(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    head = run_json(runner, tmp_path / "head.json")
    example = example_graph()
    fixed = [element for element in example.relationships if element.id != "born_in_copy"]
    monkeypatch.setattr(FakeSource, "graph", Graph(nodes=example.nodes, relationships=tuple(fixed)))
    base = run_json(runner, tmp_path / "base.json")

    result = runner.invoke(cli.app, ["compare", str(base), str(head)])
    tolerant = runner.invoke(cli.app, ["compare", "--max-drop", "0.1", str(base), str(head)])

    assert result.exit_code == 1
    assert "### fact_uniqueness: 1 new finding" in result.stdout
    # Absorb the 9.1-point drop of fact_uniqueness within the allowed drop.
    assert tolerant.exit_code == 0, tolerant.output


def test_compare_reports_invalid_reports(runner: CliRunner, tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text("{}", encoding="utf-8")

    result = runner.invoke(cli.app, ["compare", str(path), str(path)])

    assert result.exit_code == 2
    assert result.stderr.startswith(f"Error: invalid file {path}: ")
