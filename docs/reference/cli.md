# CLI

| Command                                       | Description                                      |
| --------------------------------------------- | ------------------------------------------------ |
| [`kgeval run`](#kgeval-run)                   | Evaluate a graph and report scores and findings. |
| [`kgeval compare`](#kgeval-compare)           | Compare two reports metric by metric.            |
| [`kgeval infer-schema`](#kgeval-infer-schema) | Draft a schema from the graph.                   |
| [`kgeval metrics`](#kgeval-metrics)           | List the built-in metrics.                       |

Every command accepts `--help`, and `kgeval --version` prints the installed version.

## kgeval run

```console
kgeval run [OPTIONS]
```

Evaluate a Neo4j knowledge graph and report scores and findings.

### Connection

| Option       | Default                  | Description                                                                                |
| ------------ | ------------------------ | ------------------------------------------------------------------------------------------ |
| `--uri`      | `neo4j://localhost:7687` | The connection URI. Reads `NEO4J_URI`.                                                     |
| `--username` | `neo4j`                  | The user. Reads `NEO4J_USERNAME`.                                                          |
| `--password` | None                     | The password. Reads `NEO4J_PASSWORD`. Without one, kgeval connects without authentication. |
| `--database` | The server's default     | The database to read. Reads `NEO4J_DATABASE`.                                              |

In a terminal, kgeval prompts for the URI, user, and password when they are unset, and
offers the defaults. Elsewhere, such as in CI, it uses the defaults.

### Inputs

| Option              | Default | Description                                                                                                |
| ------------------- | ------- | ---------------------------------------------------------------------------------------------------------- |
| `--schema FILE`     | None    | A [graph schema](../guide/inputs.md#graph-schema). Adds the schema metrics.                                |
| `--ground FILE`     | None    | A [ground truth](../guide/inputs.md#ground-truth). Adds precision, recall, and F1.                         |
| `--questions FILE`  | None    | [Test questions](../guide/inputs.md#test-questions). Adds `competency`.                                    |
| `--retrievals FILE` | None    | [Retrievals](../guide/inputs.md#retrievals). Adds the retrieval metrics. Requires `--questions`.           |
| `--top-k N`         | All     | How many top-ranked element IDs of each retrieval the retrieval metrics evaluate. Requires `--retrievals`. |
| `--layout FILE`     | None    | A [layout file](../guide/configuration.md#custom-layout) whose fields replace the detected ones.           |

### Judge

| Option                 | Default        | Description                                                                                                       |
| ---------------------- | -------------- | ----------------------------------------------------------------------------------------------------------------- |
| `--judge MODEL`        | None           | The [judge model](../guide/judges.md), as LiteLLM names it, such as `openai/gpt-5-mini`. Adds the judged metrics. |
| `--judge-base-url URL` | The provider's | A base URL that replaces the provider's. Requires `--judge`.                                                      |
| `--judge-cache FILE`   | None           | A SQLite file that stores verdicts for reruns. Requires `--judge`.                                                |
| `--sample-size N`      | 100            | The most units each sampled judged metric judges.                                                                 |
| `--seed N`             | 0              | The seed that selects which units are sampled.                                                                    |

### Metrics

| Option                           | Default     | Description                                                                 |
| -------------------------------- | ----------- | --------------------------------------------------------------------------- |
| `-m`, `--metric NAME`            | All enabled | Run only this metric. Repeat for several.                                   |
| `-t`, `--threshold METRIC=VALUE` | None        | Fail when the metric scores below `VALUE`, from 0 to 1. Repeat for several. |

A metric named by either option must be enabled by the inputs. For example,
`--threshold fact_recall=0.9` requires `--ground`.

### Output

| Option                 | Default         | Description                                                                                            |
| ---------------------- | --------------- | ------------------------------------------------------------------------------------------------------ |
| `-f`, `--format`       | `markdown`      | The report format: `markdown` or `json`.                                                               |
| `-o`, `--output FILE`  | Standard output | The file to write the report to.                                                                       |
| `--max-findings N`     | 10              | The most findings listed per metric in Markdown.                                                       |
| `--metadata KEY=VALUE` | None            | A description of the run, such as `model=gpt-5-mini`, that `kgeval compare` shows. Repeat for several. |
| `-v`, `--verbose`      | Off             | Log progress to standard error.                                                                        |

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
kgeval run --schema schema.yaml --ground ground_truth.yaml
kgeval run --judge openai/gpt-5-mini --judge-cache verdicts.sqlite
kgeval run -t entity_uniqueness=0.98 -t fact_provenance=0.95
kgeval run --metadata model=gpt-5-mini --format json -o report.json
```

## kgeval compare

```console
kgeval compare [OPTIONS] BASE HEAD
```

Compare the report `HEAD` with the report `BASE`, such as a baseline, metric by metric.
Both are JSON reports written by `kgeval run --format json`.

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
or a defect's details change. Findings of judged metrics are not compared, because each
run judges its own sample. Metrics whose parameters differ, such as runs against
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
kgeval run --metadata model=gpt-5-mini --format json -o base.json
# Rebuild the graph with the change under test.
kgeval run --metadata model=gpt-5 --format json -o head.json
kgeval compare base.json head.json
```

## kgeval infer-schema

```console
kgeval infer-schema [OPTIONS]
```

Write the schema the graph follows today, as YAML in neo4j-graphrag's format. Review
the draft before passing it to `--schema`, because it copies the graph's defects.

### Options

| Option                                            | Default             | Description                                            |
| ------------------------------------------------- | ------------------- | ------------------------------------------------------ |
| `--uri`, `--username`, `--password`, `--database` | As for `kgeval run` | The [connection options](#connection) of `kgeval run`. |
| `--layout FILE`                                   | None                | A layout file whose fields replace the detected ones.  |
| `-o`, `--output FILE`                             | Standard output     | The file to write the schema to.                       |
| `-v`, `--verbose`                                 | Off                 | Log progress to standard error.                        |

## kgeval metrics

```console
kgeval metrics
```

List every built-in metric by category, with the option each category needs.
