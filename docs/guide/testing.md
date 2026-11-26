# Testing with pytest

To check a graph in the same test suite as the pipeline that builds it, call
[`assert_passes`][kgeval.assert_passes] in a pytest test. Each test sets the minimum
score of its metrics, and a failing test shows the report with the findings behind each
score.

## Write a test

Load the graph once per test session in a fixture:

```python title="conftest.py"
import kgeval
import pytest
from neo4j import GraphDatabase


@pytest.fixture(scope="session")
def graph():
    with GraphDatabase.driver("<uri>", auth=("<username>", "<password>")) as driver:
        return kgeval.Neo4jSource(driver).load()
```

Then pass the graph and the metrics to check, each with a threshold, to `assert_passes`:

```python title="test_graph.py"
import kgeval


def test_entities_are_unique(graph):
    kgeval.assert_passes(graph, [kgeval.EntityUniqueness(threshold=0.95)])
```

The test fails when a metric scores below its threshold or is skipped, such as a
provenance metric on a graph without chunks. The failure message names each such metric
on its first line, which pytest's summary shows, followed by the report:

```text
E       AssertionError: entity_uniqueness scored 90.0%, below its threshold of 95.0%
E
E       # kgeval report
...
E       ### entity_uniqueness: 1 of 10 entities with a name failed
E
E       - 2 entities share the normalized name 'lee kuan yew': "Lee Kuan Yew" (Person), "lee kuan yew" (Person)
E         - nodes: `4:74f3…:4`, `4:74f3…:9`
```

`assert_passes` returns the report, so a test can check more, such as a metric's
findings. For a graph with its own labels or property keys, pass a layout to
`assert_passes` as `layout`; see [Custom layout](configuration.md#custom-layout).

## Check each metric in its own test

To see each metric pass or fail on its own, parametrize the test:

```python
import kgeval
import pytest


@pytest.mark.parametrize(
    "metric",
    [kgeval.EntityUniqueness(threshold=0.95), kgeval.FactUniqueness(threshold=1)],
    ids=lambda metric: metric.name,
)
def test_metric(graph, metric):
    kgeval.assert_passes(graph, [metric])
```

Each call evaluates the graph again, so on a large graph, checking several metrics in one
call is faster.

## Use a model

`assert_passes` takes the inputs that metrics need, such as a graph schema or a model, by
name, as `evaluate` does. Give the model a cache file, so that reruns on the same graph
reuse its verdicts and give the same scores without calling the model again; see
[Caching](models.md#caching):

```python
import kgeval

model = kgeval.Model("<model>", cache="verdicts.sqlite")


def test_facts_are_accurate(graph):
    kgeval.assert_passes(graph, [kgeval.FactAccuracy(threshold=0.9)], model=model)
```

## Check a suite

To check every metric of a [suite](suites.md) with the thresholds it sets, load it. Every
metric in the suite needs a threshold:

```python
def test_graph_meets_the_suite(graph):
    kgeval.assert_passes(graph, kgeval.load_suite("suite.yaml"), model=model)
```

## Compare test runs

Pass `--kgeval-report` to save the results of every `assert_passes` call, including the
calls that failed, as a JSON report. Then compare it with the report of an earlier run,
such as a run on the main branch, to see which metrics improved or regressed:

```console
pytest --kgeval-report head.json
kgeval compare base.json head.json
```

A report covers one graph and has one result per metric, so every call must evaluate the
same graph, and a metric checked in several tests must have the same parameters in each.
Otherwise, pytest saves no report and fails. When several tests check a metric with
different thresholds, the report keeps the highest. To read the comparison, see
[Comparing builds](comparing.md).
