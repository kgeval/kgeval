"""Test the source that loads knowledge graphs from Neo4j, wiping the integration database."""

import json
import os
import shutil
from collections import Counter
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime, time
from pathlib import Path
from types import TracebackType
from typing import Any, cast

import neo4j
import neo4j.spatial
import neo4j.time
import pytest
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

from kgeval.errors import SourceError
from kgeval.evaluation import evaluate
from kgeval.graph.model import Duration, Graph, Point, PropertyValue, canonical_json
from kgeval.report import compare
from kgeval.sources.neo4j import Neo4jSource, _convert, _read_graph
from tests.doubles import ScriptedModel
from tests.factories import EXAMPLE_DIR, example_graph

_TEST_URI = os.environ.get("KGEVAL_TEST_NEO4J_URI")

# Store one value of each property type the source converts, and embeddings it reads as sizes.
_TYPED_VALUES = """
CREATE (chunk:Chunk {text: 'Ada works at Acme.', embedding: [0.1, 0.2, 0.3]})
CREATE (ada:__Entity__:Person {
    name: 'Ada', born: date('1815-12-10'), tags: [], scores: [1, 2],
    home: point({x: 1.0, y: 2.0})
})
CREATE (acme:__Entity__:Company {name: 'Acme', age: duration('P1Y')})
CREATE (chunk)-[:HAS_ENTITY]->(ada), (chunk)-[:HAS_ENTITY]->(acme)
CREATE (ada)-[:WORKS_AT {since: 1833, weights: [0.5]}]->(acme)
"""


class FailingSession:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def __enter__(self) -> "FailingSession":
        return self

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def execute_read(self, work: Callable[..., object], *arguments: object) -> object:
        raise self.error


class FailingDriver:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def session(self, **config: object) -> FailingSession:
        return FailingSession(self.error)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Ada", "Ada"),
        (True, True),
        (3, 3),
        (2.5, 2.5),
        (neo4j.time.Date(2020, 1, 2), date(2020, 1, 2)),
        (neo4j.time.DateTime(2020, 1, 2, 3, 4, 5), datetime(2020, 1, 2, 3, 4, 5)),
        (
            neo4j.time.DateTime(2020, 1, 2, 3, 4, 5, tzinfo=UTC),
            datetime(2020, 1, 2, 3, 4, 5, tzinfo=UTC),
        ),
        (neo4j.time.Time(3, 4, 5), time(3, 4, 5)),
        (
            neo4j.time.Duration(months=14, days=3, seconds=5, nanoseconds=7),
            Duration(months=14, days=3, seconds=5, nanoseconds=7),
        ),
        (neo4j.spatial.CartesianPoint((1.0, 2.0)), Point(srid=7203, coordinates=(1.0, 2.0))),
        ([neo4j.time.Date(2020, 1, 2)], [date(2020, 1, 2)]),
        ([], []),
    ],
)
def test_driver_values_convert_to_property_values(value: object, expected: PropertyValue) -> None:
    converted = _convert(value)

    assert converted == expected
    assert type(converted) is type(expected)


@pytest.mark.parametrize("value", [b"bytes", ["a", b"b"], None])
def test_unsupported_driver_values_are_dropped(value: object) -> None:
    assert _convert(value) is None


class RecordedTransaction:
    def __init__(self, *results: list[dict[str, Any]]) -> None:
        self.results = iter(results)

    def run(self, query: str) -> list[dict[str, Any]]:
        return next(self.results)


def decoded(text: str) -> str:
    # Build a new string object, as the driver does for every record.
    return "".join(list(text))


def test_loaded_elements_share_repeated_strings() -> None:
    nodes = [
        {
            "id": decoded(node_id),
            "labels": [decoded("Person")],
            "properties": [[decoded("name"), node_id]],
            "embedding_dimensions": [[decoded("embedding"), 3]],
        }
        for node_id in ("ada", "bob")
    ]
    relationships = [
        {
            "id": relationship_id,
            "type": decoded("KNOWS"),
            "source_id": decoded("ada"),
            "target_id": decoded("bob"),
            "properties": [[decoded("name"), relationship_id]],
            "embedding_dimensions": [],
        }
        for relationship_id in ("knows_1", "knows_2")
    ]

    (ada, bob), (first, second) = _read_graph(
        cast(neo4j.ManagedTransaction, RecordedTransaction(nodes, relationships))
    )

    assert first.source_id is ada.id
    assert first.target_id is bob.id
    assert first.type is second.type
    assert ada.labels[0] is bob.labels[0]
    assert next(iter(ada.properties)) is next(iter(first.properties))
    assert next(iter(ada.embedding_dimensions)) is next(iter(bob.embedding_dimensions))


class PredicateSyntaxError(Neo4jError):
    """Stand in for the error a pre-5.10 server raises on list type predicates."""

    code = "Neo.ClientError.Statement.SyntaxError"
    message = "Invalid input 'IS'"


def test_load_names_the_minimum_version_when_the_server_rejects_the_queries() -> None:
    source = Neo4jSource(cast(neo4j.Driver, FailingDriver(PredicateSyntaxError())))

    with pytest.raises(SourceError, match=r"kgeval needs Neo4j 5.10 or later"):
        source.load()


def test_load_wraps_other_driver_errors_without_the_version_hint() -> None:
    error = neo4j.exceptions.ServiceUnavailable("connection refused")
    source = Neo4jSource(cast(neo4j.Driver, FailingDriver(error)))

    with pytest.raises(SourceError, match="connection refused") as raised:
        source.load()
    assert "5.10" not in str(raised.value)


@pytest.fixture
def driver() -> Iterator[neo4j.Driver]:
    if _TEST_URI is None:
        pytest.skip("set KGEVAL_TEST_NEO4J_URI to a disposable Neo4j database")
    password = os.environ.get("KGEVAL_TEST_NEO4J_PASSWORD")
    username = os.environ.get("KGEVAL_TEST_NEO4J_USERNAME", "neo4j")
    auth = (username, password) if password else None
    with GraphDatabase.driver(_TEST_URI, auth=auth) as driver, driver.session() as session:
        session.run("MATCH (node) DETACH DELETE node").consume()
        yield driver
        session.run("MATCH (node) DETACH DELETE node").consume()


@pytest.fixture
def source(driver: neo4j.Driver) -> Neo4jSource:
    with driver.session() as session:
        session.run(_TYPED_VALUES).consume()
    return Neo4jSource(driver)


@pytest.fixture
def example_source(driver: neo4j.Driver) -> Neo4jSource:
    with driver.session() as session:
        session.run((EXAMPLE_DIR / "seed.cypher").read_text(encoding="utf-8")).consume()
    return Neo4jSource(driver)


def content(graph: Graph) -> tuple[Counter[str], Counter[tuple[str, str, str, str]]]:
    """Return a graph's nodes and relationships by content, without element IDs."""
    nodes = {
        node.id: canonical_json([node.labels, node.properties, node.embedding_dimensions])
        for node in graph.nodes
    }
    relationships = Counter(
        (
            nodes[link.source_id],
            link.type,
            nodes[link.target_id],
            canonical_json([link.properties, link.embedding_dimensions]),
        )
        for link in graph.relationships
    )
    return Counter(nodes.values()), relationships


@pytest.mark.integration
def test_load_reads_properties_and_the_dimensions_of_embeddings(source: Neo4jSource) -> None:
    graph = source.load()

    by_label = {node.labels[0]: node for node in graph.nodes}
    assert by_label["Chunk"].properties == {"text": "Ada works at Acme."}
    assert by_label["Chunk"].embedding_dimensions == {"embedding": 3}
    assert by_label["Person"].embedding_dimensions == {}
    assert by_label["Person"].properties == {
        "name": "Ada",
        "born": date(1815, 12, 10),
        "tags": [],
        "scores": [1, 2],
        "home": Point(srid=7203, coordinates=(1.0, 2.0)),
    }
    assert by_label["Company"].properties["age"] == Duration(months=12)
    [works_at] = [link for link in graph.relationships if link.type == "WORKS_AT"]
    assert (works_at.properties, works_at.embedding_dimensions) == ({"since": 1833}, {"weights": 1})


@pytest.mark.integration
def test_example_seed_creates_the_example_graph_of_the_unit_tests(
    example_source: Neo4jSource,
) -> None:
    assert content(example_source.load()) == content(example_graph())


@pytest.mark.integration
def test_example_notebook_loads_and_evaluates_the_example_graph(
    driver: neo4j.Driver,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    # Run a copy of the example on the test database, so its verdict cache stays out of the repo.
    monkeypatch.setenv("NEO4J_URI", cast(str, _TEST_URI))
    for setting in ("USERNAME", "PASSWORD"):
        if value := os.environ.get(f"KGEVAL_TEST_NEO4J_{setting}"):
            monkeypatch.setenv(f"NEO4J_{setting}", value)
        else:
            monkeypatch.delenv(f"NEO4J_{setting}", raising=False)
    shutil.copytree(EXAMPLE_DIR, tmp_path, dirs_exist_ok=True)
    monkeypatch.chdir(tmp_path)
    model = ScriptedModel()
    monkeypatch.setattr("kgeval.Model", lambda name, **options: model)
    notebook = json.loads((tmp_path / "evaluate.ipynb").read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {}

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec("".join(cell["source"]), namespace)

    output = capsys.readouterr().out
    assert "2 entities share the normalized name 'lee kuan yew'" in output
    assert "label: PERSON" in output
    born = {
        element.id
        for element in (*namespace["graph"].nodes, *namespace["graph"].relationships)
        if element.id in namespace["retrievals"]["Where was Lee Kuan Yew born?"]
    }
    assert len(born) == len(namespace["retrievals"]["Where was Lee Kuan Yew born?"]) > 0
    assert model.prompts
    assert namespace["check"].passed is False
    assert not compare(namespace["report"], evaluate(namespace["fixed"])).regressed
