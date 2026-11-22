# kgeval

[![PyPI](https://img.shields.io/pypi/v/kgeval)](https://pypi.org/project/kgeval/)
[![Python](https://img.shields.io/pypi/pyversions/kgeval)](https://pypi.org/project/kgeval/)
[![CI](https://github.com/kgeval/kgeval/actions/workflows/ci.yml/badge.svg)](https://github.com/kgeval/kgeval/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/kgeval/kgeval)](https://github.com/kgeval/kgeval/blob/main/LICENSE)

kgeval is an evaluation framework for knowledge graphs.

## Features

- Score a graph on 29 built-in metrics
- Trace every finding to the nodes and relationships involved
- Run reference-free and reference-based evaluation
- Compare evaluation runs over time

## Requirements

- Python 3.11+
- Neo4j 5.10+ or Aura

## Installation

```console
pip install kgeval
```

## Usage

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

## Metrics

| Input          | Evaluates                                                               |
| -------------- | ----------------------------------------------------------------------- |
| Graph only     | Uniqueness, connectivity, naming, provenance, consistency, and coverage |
| Graph schema   | Conformance and completeness                                            |
| Ground truth   | Graph precision, recall, and F1                                         |
| Test questions | Answerability                                                           |
| Retrievals     | Retriever precision and recall                                          |
| LLM judge      | Accuracy, resolution, disambiguation, and extraction completeness       |

See the [metrics guide](https://kgeval.github.io/kgeval/guide/metrics/) for the full list.

## Documentation

For the full guide, see the [documentation](https://kgeval.github.io/kgeval/). For a
complete walkthrough, see the [example notebook](https://github.com/kgeval/kgeval/blob/main/examples/singapore/evaluate.ipynb).
