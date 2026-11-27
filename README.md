# kgeval

[![PyPI](https://img.shields.io/pypi/v/kgeval)](https://pypi.org/project/kgeval/)
[![Python](https://img.shields.io/pypi/pyversions/kgeval)](https://pypi.org/project/kgeval/)
[![CI](https://github.com/kgeval/kgeval/actions/workflows/ci.yml/badge.svg)](https://github.com/kgeval/kgeval/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/kgeval/kgeval)](https://github.com/kgeval/kgeval/blob/main/LICENSE)

kgeval is an evaluation framework for knowledge graphs.

It measures the quality of the knowledge graphs behind GraphRAG and other
retrieval-augmented generation (RAG) applications, and reports every defect with the nodes
and relationships involved. It reads Neo4j graphs from LLM Graph Builder, neo4j-graphrag,
LangChain, LlamaIndex, Microsoft GraphRAG, and Graphiti without configuration, and scores
them with deterministic metrics and with an LLM as a judge.

## Features

- Score a graph on 29 built-in metrics
- Define custom metrics in one sentence
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

## Metrics

| Metrics   | Evaluate                                                                |
| --------- | ----------------------------------------------------------------------- |
| Intrinsic | Uniqueness, connectivity, naming, provenance, consistency, and coverage |
| Schema    | Conformance and completeness                                            |
| Reference | Graph precision, recall, F1, and answerability                          |
| Model     | Accuracy, resolution, disambiguation, and extraction completeness       |
| Retrieval | Retriever precision and recall                                          |

See the [metrics guide](https://kgeval.github.io/kgeval/guide/metrics/) for the full list.

## Documentation

For the full guide, see the [documentation](https://kgeval.github.io/kgeval/). For a
complete walkthrough, see the [example notebook](https://github.com/kgeval/kgeval/blob/main/examples/singapore/evaluate.ipynb).
