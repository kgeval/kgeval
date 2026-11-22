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
from tests.doubles import ScriptedJudge
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
    answers = iter(["neo4j+s://db.example", "admin", "secret"])
    monkeypatch.setattr(typer, "prompt", lambda *args, **kwargs: next(answers))

    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    with cli._connect(None, None, None):
        pass
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: False))
    with cli._connect(None, None, None):
        pass

    assert connected == [
        ("neo4j+s://db.example", ("admin", "secret")),
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


def test_run_adds_a_metric_suite_for_each_example_input(runner: CliRunner, tmp_path: Path) -> None:
    retrievals = tmp_path / "retrievals.json"
    retrievals.write_text(json.dumps(EXAMPLE_RETRIEVALS), encoding="utf-8")

    result = runner.invoke(
        cli.app,
        [
            "run",
            *("--ground", str(EXAMPLE_DIR / "ground_truth.yaml")),
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
        (["-m", "fact_accuracy"], "fact_accuracy needs --judge"),
        (["-m", "retrieval_recall"], "retrieval_recall needs --questions and --retrievals"),
        (["--retrievals", str(EXAMPLE_DIR / "questions.yaml")], "needs --questions"),
        (["--judge-base-url", "http://localhost:8000/v1"], "needs --judge"),
        (["--judge-cache", "verdicts.sqlite"], "needs --judge"),
        (["--top-k", "5"], "needs --retrievals"),
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


def test_run_adds_the_judged_metrics_for_a_judge(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[tuple[str, str | None]] = []

    def create(model: str, *, base_url: str | None = None) -> ScriptedJudge:
        created.append((model, base_url))
        return ScriptedJudge()

    monkeypatch.setattr(cli, "create_judge", create)

    result = runner.invoke(
        cli.app,
        ["run", "--judge", "groq:test-model", "-m", "fact_accuracy", "-m", "node_type_accuracy"],
    )

    assert result.exit_code == 0, result.output
    assert created == [("groq:test-model", None)]
    assert "| fact_accuracy | 100.0% |" in result.stdout
    assert "| node_type_accuracy | 100.0% |" in result.stdout


def test_run_names_a_missing_judge_key(runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    result = runner.invoke(cli.app, ["run", "--judge", "groq:test-model"])

    assert result.exit_code == 2
    assert result.stderr == "Error: missing environment variable: GROQ_API_KEY\n"


def test_version_prints_the_installed_version(runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["--version"])

    assert result.exit_code == 0
    assert result.output == f"kgeval {version('kgeval')}\n"


def test_metrics_lists_every_builtin_metric_by_category(runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["metrics"])

    assert result.exit_code == 0, result.output
    assert "## intrinsic (runs by default)" in result.stdout
    assert "## reference (needs --ground or --questions)" in result.stdout
    assert "## judged (needs --judge)" in result.stdout
    assert "## retrieval (needs --questions and --retrievals)" in result.stdout
    assert all(f"| {metric_type.name} |" in result.stdout for metric_type in BUILTIN_METRICS)


def test_infer_schema_writes_a_schema_that_loads_back(runner: CliRunner, tmp_path: Path) -> None:
    path = tmp_path / "schema.yaml"

    result = runner.invoke(cli.app, ["infer-schema", "--output", str(path)])

    assert result.exit_code == 0, result.output
    schema = load_schema(path)
    assert [node_type.label for node_type in schema.node_types][:2] == ["Person", "Organization"]


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
    assert result.stderr.startswith(f"Error: invalid report {path}: ")
