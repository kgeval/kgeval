# Quickstart

This page evaluates a knowledge graph in Neo4j, walks through the report, and shows how
to enable more metrics.

## Evaluate a graph

kgeval reads the whole graph into memory through a read-only session and evaluates
that snapshot.

=== "CLI"

    ```console
    kgeval run
    ```

    kgeval prompts for the connection, or reads it from `NEO4J_URI`, `NEO4J_USERNAME`,
    and `NEO4J_PASSWORD`.

=== "Python"

    ```python
    import kgeval
    from neo4j import GraphDatabase

    with GraphDatabase.driver("<uri>", auth=("<username>", "<password>")) as driver:
        graph = kgeval.Neo4jSource(driver).load()

    report = kgeval.evaluate(graph)
    print(report.to_markdown())
    ```

    In a notebook, a report renders itself as Markdown.

With no other inputs, kgeval runs the ten intrinsic metrics, which need only the graph.

## Read the report

The examples below come from the
[example notebook](https://github.com/kgeval/kgeval/blob/main/examples/singapore/evaluate.ipynb),
with element IDs shortened.

### Summary

The report opens with the size of the graph and how kgeval read it:

```text
- Graph: 19 nodes (11 entities, 7 lexical nodes, 1 community node, 0 other nodes) and 40 relationships (11 facts, 14 provenance relationships)
- Layout: lexical labels Chunk, Document; community labels __Community__; base labels __Entity__; names from name, title, id
- Reproduce with kgeval 0.1.0 on graph `sha256:c4a8…f268`
```

If the counts look wrong, check the layout first. kgeval detects it from the graph's
labels, and [Configuration](guide/configuration.md#graph-layout) explains how to
change it.

### Scores

Each metric gets one row:

| Metric              | Score |                      Failed | Threshold | Result |
| ------------------- | ----: | --------------------------: | --------: | ------ |
| entity_uniqueness   | 90.0% | 1 / 10 entities with a name |         — |        |
| entity_naming       | 81.8% |             2 / 11 entities |         — |        |
| extraction_coverage | 80.0% |                1 / 5 chunks |         — |        |

- **Score** is the share of units that pass, so 100% is perfect.
- **Failed** counts the units that failed out of those evaluated, in the metric's own
  unit.
- **Threshold** and **Result** are filled in for metrics with a threshold, set with
  [`--threshold`](reference/cli.md#kgeval-run).

A metric with nothing to evaluate is skipped and shows the reason instead of a score.
For example, the provenance metrics skip a graph without chunks.

### Findings

Each finding describes one defect and lists the elements involved:

```text
### entity_uniqueness: 1 of 10 entities with a name failed

- 2 entities share the normalized name 'lee kuan yew': "Lee Kuan Yew" (Person), "lee kuan yew" (Person)
  - nodes: `4:a63a…:4`, `4:a63a…:9`
```

To inspect those nodes in Neo4j Browser:

```cypher
MATCH (n) WHERE elementId(n) IN ['4:a63a…:4', '4:a63a…:9'] RETURN n
```

The Markdown report lists up to 10 findings per metric, and `--max-findings` changes
that limit. The JSON report and the Python `Report` hold all of them.

## Enable more metrics

Each input enables another set of metrics:

| Input          | Option         | Enables                                         |
| -------------- | -------------- | ----------------------------------------------- |
| Graph schema   | `--schema`     | Conformance and completeness against the schema |
| Ground truth   | `--ground`     | Fact and entity precision, recall, and F1       |
| Test questions | `--questions`  | `competency`                                    |
| Retrievals     | `--retrievals` | Retrieval recall and precision                  |
| LLM judge      | `--judge`      | Six judged metrics                              |

```console
kgeval run --schema schema.yaml --ground ground_truth.yaml --questions questions.yaml
```

[Inputs](guide/inputs.md) describes each file format, and
[LLM judges](guide/judges.md) covers choosing a model.

## Try the example notebook

The [example notebook](https://github.com/kgeval/kgeval/blob/main/examples/singapore/evaluate.ipynb)
evaluates a small graph shaped like LLM Graph Builder output, with one defect planted for
every metric, and walks through every feature.
