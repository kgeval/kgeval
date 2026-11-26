# kgeval

kgeval is an evaluation framework for knowledge graphs.

It scores a graph on 29 metrics, covering duplicate entities, conformance to a graph
schema, agreement with a ground truth, and whether each fact is supported by the text it
was extracted from, and reports every defect with the nodes and relationships involved.
Comparing the reports of two versions of a graph shows which metrics improved or
regressed.

## Features

- Score a graph on 29 built-in metrics
- Trace every finding to the nodes and relationships involved
- Run reference-free and reference-based evaluation
- Compare evaluation runs over time
- Unit test graph quality in CI with pytest

## Requirements

- Python 3.11+
- Neo4j 5.10+ or Aura, or a [graph snapshot](guide/configuration.md#snapshots)

## Installation

```console
pip install kgeval
```

## Next steps

The [quickstart](quickstart.md) evaluates a graph and reads the report, and the
[metrics guide](guide/metrics.md) describes what each metric checks and what it needs.
For a complete walkthrough on a graph with one planted defect for every metric, see the
[example notebook](https://github.com/kgeval/kgeval/blob/main/examples/singapore/evaluate.ipynb).
