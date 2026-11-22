# Comparing builds

To test a change to how a graph is built, such as the extraction prompt, the model, or
the schema, compare a report of each build to see whether the change helped. Build both
graphs from the same documents, so that the comparison measures the pipeline rather
than the corpus.

## Compare two builds

Describe each run with `--metadata`, write its report as JSON, and compare the two:

=== "CLI"

    ```console
    kgeval run --metadata model=gpt-5-mini --format json -o base.json
    # Rebuild the graph with the change under test.
    kgeval run --metadata model=gpt-5 --format json -o head.json
    kgeval compare base.json head.json
    ```

=== "Python"

    ```python
    base = kgeval.evaluate(base_graph, metrics, metadata={"model": "gpt-5-mini"})
    head = kgeval.evaluate(head_graph, metrics, metadata={"model": "gpt-5"})
    comparison = kgeval.compare(base, head)
    ```

Run both with the same inputs and options, so that each metric is scored the same way
in both reports. A metric whose parameters differ, such as one scored against another
ground truth, is marked "parameters changed, so not compared", since its scores measure
different things. The comparison also warns when the reports come from different kgeval
versions or read their graphs with different layouts.

## Read the comparison

The comparison opens with a summary of how the scores moved, followed by the metadata
that differs between the two runs:

```text
## Summary

Scores: 9 improved, 2 worsened, 16 unchanged, 1 inconclusive

| Metadata | Base | Head |
| --- | --- | --- |
| model | gpt-5-mini | gpt-5 |
```

Then each metric gets one row:

| Metric        |         Base |         Head |                               Change | New | Resolved | Note                                                      |
| ------------- | -----------: | -----------: | -----------------------------------: | --: | -------: | --------------------------------------------------------- |
| fact_accuracy | 88.0% of 100 | 85.0% of 100 | -3.0 pp (95% CI -12.7 pp to +6.7 pp) |   — |        — | inconclusive; findings of judged metrics are not compared |

- **Base** and **Head** show each score with the number of units it counts.
- **Change** is in percentage points. For a sampled score, it includes a 95% interval.
- **New** and **Resolved** count the findings that only the head or only the base has,
  matched by subject: the unit at fault, named by content, such as an entity's name and
  node types. A finding whose details change, such as a group of duplicates that grows,
  keeps its subject.

The verdict at the top, REGRESSED or NO REGRESSION, says whether any metric scores worse
in the head, beyond sampling noise and `--max-drop`, with a new defect, or is no longer
scored. Rebuilding a graph with an LLM moves its scores a little even when nothing
changed, so give such comparisons an allowed drop, such as `--max-drop 0.01`. See
[`kgeval compare`](../reference/cli.md#kgeval-compare).

## Choose which metrics to trust

Not all metrics compare the same way across builds:

- **Metrics over fixed inputs compare like for like.** Test questions and ground truth
  facts are the same in both builds, and so are chunks, as long as both builds split
  the documents the same way. Start with `competency`, `retrieval_recall`,
  `fact_recall`, `entity_recall`, `extraction_coverage`, and `extraction_completeness`.
- **Metrics over extracted entities and facts need context.** Each build extracts its
  own, so a score such as `fact_accuracy` or `fact_precision` counts a different
  population in each. A build can raise precision just by extracting less, so read
  precision together with recall, and check the unit counts.
- **New and resolved findings mostly reflect churn when the extraction changes**,
  because a different build names and connects things differently. They are precise
  for a rebuild with the same pipeline, and for metrics over fixed inputs.

## Account for sampling noise

Judged metrics score a sample, so two builds can differ by chance alone. For each
sampled score, the comparison shows a 95% interval for the change, and marks the change
**inconclusive** when that interval includes zero; an inconclusive change is no
regression. Deterministic scores have no sampling noise, so any change in them is real.

To detect smaller differences, judge larger samples with
[`--sample-size`](judges.md#sampling). Keep `--seed` the same in both runs: kgeval
picks samples by content, so both builds then judge the units they have in common.

kgeval computes the interval with Newcombe's method from each score's Wilson interval.
It treats the two samples as independent, which is conservative when they share units.
