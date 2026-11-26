# CLI

| Command                                       | Description                                      |
| --------------------------------------------- | ------------------------------------------------ |
| [`kgeval run`](#kgeval-run)                   | Evaluate a graph and report scores and findings. |
| [`kgeval compare`](#kgeval-compare)           | Compare two reports metric by metric.            |
| [`kgeval infer-schema`](#kgeval-infer-schema) | Draft a schema from the graph.                   |
| [`kgeval metrics`](#kgeval-metrics)           | List the built-in metrics and their inputs.      |

Every command accepts `--help`, and `kgeval --version` prints the installed version.

## kgeval run

```console
kgeval run [OPTIONS]
```

Evaluate a Neo4j knowledge graph and report scores and findings. Without `--metric` or
`--suite`, kgeval runs every metric whose inputs are given.

### Graph

| Option          | Default                  | Description                                                                                      |
| --------------- | ------------------------ | ------------------------------------------------------------------------------------------------ |
| `--uri`         | `neo4j://localhost:7687` | The connection URI. Reads `NEO4J_URI`.                                                           |
| `--username`    | `neo4j`                  | The user. Reads `NEO4J_USERNAME`.                                                                |
| `--password`    | None                     | The password. Reads `NEO4J_PASSWORD`. Without one, kgeval connects without authentication.       |
| `--database`    | The server's default     | The database to read. Reads `NEO4J_DATABASE`.                                                    |
| `--layout FILE` | Detected                 | A [layout file](../guide/configuration.md#custom-layout) whose fields replace the detected ones. |

In a terminal, kgeval prompts for the URI, user, and password when they are unset, and
offers the defaults. Elsewhere, such as in CI, it uses the defaults.

### Inputs

| Option                | Default | Description                                                                                 |
| --------------------- | ------- | ------------------------------------------------------------------------------------------- |
| `--schema FILE`       | None    | A [graph schema](../guide/inputs.md#graph-schema).                                          |
| `--ground-truth FILE` | None    | A [ground truth](../guide/inputs.md#ground-truth).                                          |
| `--questions FILE`    | None    | [Test questions](../guide/inputs.md#test-questions).                                        |
| `--retrievals FILE`   | None    | [Retrievals](../guide/inputs.md#retrievals) for the test questions. Requires `--questions`. |

### Model

| Option                 | Default        | Description                                                                        |
| ---------------------- | -------------- | ---------------------------------------------------------------------------------- |
| `--model MODEL`        | None           | The [model](../guide/models.md), as LiteLLM names it, such as `openai/gpt-5-mini`. |
| `--model-base-url URL` | The provider's | A base URL that replaces the provider's. Requires `--model`.                       |
| `--model-cache FILE`   | None           | A SQLite file that stores verdicts for reruns. Requires `--model`.                 |

### Metrics

| Option                           | Default | Description                                                                                                  |
| -------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------ |
| `-m`, `--metric NAME`            | None    | Run only this metric. Repeat for several.                                                                    |
| `--suite FILE`                   | None    | A [suite](../guide/suites.md) of the metrics to run, with their settings.                                    |
| `-t`, `--threshold METRIC=VALUE` | None    | Fail when the metric scores below `VALUE`, from 0 to 1. Repeat for several.                                  |
| `--sample-size N`                | 100     | The most units each sampled metric checks, unless the suite sets it.                                         |
| `--seed N`                       | 0       | The seed that selects which units are sampled, unless the suite sets it.                                     |
| `--top-k N`                      | All     | How many top-ranked element IDs of each retrieval the retrieval metrics evaluate, unless the suite sets `k`. |

A metric named by `--metric` or `--threshold` must have its inputs given, and must be in
the suite when there is one. For example, `--threshold fact_recall=0.9` requires
`--ground-truth`. `kgeval metrics` lists the inputs each metric needs.

### Output

| Option                 | Default         | Description                                                                                          |
| ---------------------- | --------------- | ---------------------------------------------------------------------------------------------------- |
| `-f`, `--format`       | `markdown`      | The report format: `markdown` or `json`.                                                             |
| `-o`, `--output FILE`  | Standard output | The file to write the report to.                                                                     |
| `--max-findings N`     | 10              | The most findings listed per metric in Markdown.                                                     |
| `--metadata KEY=VALUE` | None            | A description of the run, such as `chunk_size=512`, that `kgeval compare` shows. Repeat for several. |
| `-v`, `--verbose`      | Off             | Log progress to standard error.                                                                      |

In a terminal, the Markdown report is rendered. When written to a file or a pipe, it
stays plain Markdown. The JSON report holds every field of
[`Report`][kgeval.Report], with all findings.

### Exit status

| Status | Meaning                                                         |
| ------ | --------------------------------------------------------------- |
| 0      | Every metric with a threshold met it.                           |
| 1      | A metric with a threshold scored below it or was skipped.       |
| 2      | An error, such as an invalid input file or a failed connection. |

### Examples

```console
kgeval run --schema schema.yaml --ground-truth ground_truth.yaml
kgeval run --model <model> --model-cache verdicts.sqlite
kgeval run --model <model> --suite suite.yaml
kgeval run -t entity_uniqueness=0.98 -t fact_provenance=0.95
kgeval run --metadata chunk_size=512 --format json -o report.json
```

## kgeval compare

```console
kgeval compare [OPTIONS] BASE HEAD
```

Compare the report `HEAD` with the report `BASE`, such as a baseline, metric by metric.
Both are JSON reports written by `kgeval run --format json` or `pytest --kgeval-report`.

A metric regresses when `HEAD` scores it worse than `BASE`, or no longer scores a
metric that `BASE` scored, because it skips the metric or does not run it. `HEAD`
scores worse when its score is lower by more than `--max-drop` and, for a sampled score,
by more than sampling explains. For a deterministic metric, `HEAD` must also have a new
finding or more failed units, so that a score that drops only because the population
shrank, as when duplicates that passed are removed, is no regression. New findings
without a lower score are no regression either: a rebuild fixes some defects as it adds
others.

Findings are matched by their subject, which names the unit at fault by content rather
than by element ID or wording, so a baseline still matches after the graph is rebuilt
or a defect's details change. Findings of model metrics are not compared, because each
run checks its own sample. Metrics whose parameters differ, such as runs against
another ground truth, are not compared at all.

The comparison opens with a summary: how many scores improved, worsened, stayed the
same, or changed by less than sampling noise, and the metadata that differs.
[Comparing builds](../guide/comparing.md) explains how to read it.

### Options

| Option                | Default         | Description                                                                                                                          |
| --------------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `--max-drop VALUE`    | 0               | The largest score drop, from 0 to 1, that is not a regression, such as `0.01` to absorb the noise of rebuilding a graph with an LLM. |
| `-o`, `--output FILE` | Standard output | The file to write the comparison to.                                                                                                 |
| `--max-findings N`    | 10              | The most new and resolved findings listed per metric.                                                                                |

### Exit status

| Status | Meaning                                 |
| ------ | --------------------------------------- |
| 0      | No metric regressed.                    |
| 1      | A metric regressed.                     |
| 2      | An error, such as an unreadable report. |

### Example

```console
kgeval run --metadata chunk_size=512 --format json -o base.json
# Rebuild the graph with the change under test.
kgeval run --metadata chunk_size=1024 --format json -o head.json
kgeval compare base.json head.json
```

## kgeval infer-schema

```console
kgeval infer-schema [OPTIONS]
```

Write the schema the graph follows today, as YAML in neo4j-graphrag's format. Review
the draft before passing it to `--schema`, because it copies the graph's defects.

### Options

| Option                                            | Default             | Description                                           |
| ------------------------------------------------- | ------------------- | ----------------------------------------------------- |
| `--uri`, `--username`, `--password`, `--database` | As for `kgeval run` | The [graph options](#graph) of `kgeval run`.          |
| `--layout FILE`                                   | None                | A layout file whose fields replace the detected ones. |
| `-o`, `--output FILE`                             | Standard output     | The file to write the schema to.                      |
| `-v`, `--verbose`                                 | Off                 | Log progress to standard error.                       |

## kgeval metrics

```console
kgeval metrics
```

List every built-in metric by category, with the options that give the inputs it needs.
