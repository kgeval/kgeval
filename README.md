# kgeval

[![PyPI](https://img.shields.io/pypi/v/kgeval)](https://pypi.org/project/kgeval/)
[![Python](https://img.shields.io/pypi/pyversions/kgeval)](https://pypi.org/project/kgeval/)
[![CI](https://github.com/kgeval/kgeval/actions/workflows/ci.yml/badge.svg)](https://github.com/kgeval/kgeval/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/kgeval/kgeval)](https://github.com/kgeval/kgeval/blob/main/LICENSE)

kgeval is an evaluation framework for knowledge graphs.

It scores a graph on 29 metrics, covering duplicate entities, conformance to a graph
schema, agreement with a ground truth, and whether each fact is supported by the text it
was extracted from, and reports every defect with the nodes and relationships involved.
Comparing the reports of two versions of a graph shows which metrics improved or
regressed.

## Features

- Score a graph on 29 built-in metrics
- Define metrics of your own in one sentence
- Trace every finding to the nodes and relationships involved
- Run reference-free and reference-based evaluation
- Compare evaluation runs over time
- Unit test graph quality in CI with pytest

## Requirements

- Python 3.11+
- Neo4j 5.10+ or Aura, or a [graph snapshot](https://kgeval.github.io/kgeval/guide/configuration/#snapshots)

## Installation

```console
pip install kgeval
```

## Usage

kgeval runs from the command line, from Python, or in pytest. Each loads a snapshot of the
graph, evaluates it in memory, and produces a report of scores and findings.

### CLI

```console
kgeval run
```

### Python

```python
import kgeval
from neo4j import GraphDatabase

with GraphDatabase.driver("<uri>", auth=("<username>", "<password>")) as driver:
    graph = kgeval.Neo4jSource(driver).load()

report = kgeval.evaluate(graph)
print(report.to_markdown())
```

### pytest

```python
import kgeval
from neo4j import GraphDatabase


def test_entities_are_unique():
    with GraphDatabase.driver("<uri>", auth=("<username>", "<password>")) as driver:
        graph = kgeval.Neo4jSource(driver).load()

    kgeval.assert_passes(graph, [kgeval.EntityUniqueness(threshold=0.95)])
```

## Report

The report gives each metric a score, the share of units such as entities or facts that
pass, so 100% is perfect. Each finding describes one defect and lists the IDs of the nodes
and relationships involved, so you can find them in the graph:

```text
### entity_uniqueness: 1 of 10 entities with a name failed

- 2 entities share the normalized name 'lee kuan yew': "Lee Kuan Yew" (Person), "lee kuan yew" (Person)
  - nodes: `4:9e643a2e-be5f-4bd0-8757-2f90968f1f8a:4`, `4:9e643a2e-be5f-4bd0-8757-2f90968f1f8a:9`
```

## Metrics

kgeval has 29 intrinsic, schema, reference, model, and retrieval metrics. It runs every
metric whose inputs you give. The intrinsic metrics need only the graph, and the others
need an input, such as a schema or a model:

| Input          | Evaluates                                                               |
| -------------- | ----------------------------------------------------------------------- |
| Graph only     | Uniqueness, connectivity, naming, provenance, consistency, and coverage |
| Graph schema   | Conformance and completeness                                            |
| Ground truth   | Graph precision, recall, and F1                                         |
| Test questions | Answerability                                                           |
| Retrievals     | Retriever precision and recall                                          |
| Model          | Accuracy, resolution, disambiguation, and extraction completeness       |

See the [metrics guide](https://kgeval.github.io/kgeval/guide/metrics/) for the full list.

## Documentation

For the full guide, see the [documentation](https://kgeval.github.io/kgeval/). For a
complete walkthrough, see the [example notebook](https://github.com/kgeval/kgeval/blob/main/examples/singapore/evaluate.ipynb).
