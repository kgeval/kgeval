# Suites

A suite lists the metrics to run and their settings, in a YAML or JSON file. Without a
suite, kgeval runs every metric whose inputs you give. With a suite, it runs exactly the
metrics the suite lists, in that order.

```yaml title="suite.yaml"
entity_uniqueness:
  threshold: 0.95
fact_provenance:
fact_accuracy:
  threshold: 0.9
  sample_size: 200
```

=== "CLI"

    ```console
    kgeval run --model <model> --suite suite.yaml
    ```

=== "Python"

    ```python
    model = kgeval.Model("<model>")
    metrics = kgeval.load_suite("suite.yaml")
    report = kgeval.evaluate(graph, metrics, model=model)
    ```

A suite holds no inputs, such as a graph schema or a model. They come from the command
line, or from the arguments of [`evaluate`][kgeval.evaluate], so one suite serves every
graph and model. A suite that lists a metric whose input is not given, such as
`fact_accuracy` without a model, is rejected before any metric runs. To run only some
metrics of a suite, name them with `--metric`.

## Settings

A metric takes its parameters as settings. A metric without settings needs only its
name, as in `fact_provenance:`.

| Setting       | Applies to            | Description                                                    |
| ------------- | --------------------- | -------------------------------------------------------------- |
| `threshold`   | Every metric          | The minimum passing score, from 0 to 1.                        |
| `sample_size` | Sampled model metrics | The most units the metric checks.                              |
| `seed`        | Sampled model metrics | The seed that selects which units are sampled.                 |
| `concurrency` | Model metrics         | How many requests to the model run at once.                    |
| `k`           | Retrieval metrics     | How many top-ranked element IDs of each retrieval to evaluate. |

The sampled model metrics are the [model metrics](metrics.md#model-metrics) other than
`vocabulary_resolution`. A setting in the suite takes precedence over `--sample-size`,
`--seed`, and `--top-k`, and `--threshold` takes precedence over the suite.
