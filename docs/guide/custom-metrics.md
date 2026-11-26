# Custom metrics

A custom metric checks a quality that no built-in metric covers, such as a rule of your
domain. Pass it to [`evaluate`][kgeval.evaluate] or
[`assert_passes`][kgeval.assert_passes] as you would a built-in metric, and its result
appears in the report and in comparisons.

## Write a metric

Subclass [`Metric`][kgeval.Metric] and implement `evaluate`, which checks each unit of
the metric's population, such as each person, and returns the counts with a finding for
each unit that fails:

```python
import kgeval


class BirthplaceCompleteness(kgeval.Metric):
    """Check that every person has a birthplace."""

    name = "birthplace_completeness"
    category = kgeval.MetricCategory.INTRINSIC
    unit = "people"

    def evaluate(self, graph: kgeval.GraphView, inputs: kgeval.Inputs) -> kgeval.MetricResult:
        people = [
            entity for entity in graph.entities if "Person" in graph.layout.node_types(entity)
        ]
        born = {
            fact.source_id for fact in graph.facts if graph.relationship_type(fact) == "BORN_IN"
        }
        findings = [
            kgeval.Finding.about(
                person,
                f"{graph.describe(person)} has no birthplace",
                subject=graph.describe(person),
            )
            for person in people
            if person.id not in born
        ]
        return self.result(inputs, len(people), len(findings), findings)


report = kgeval.evaluate(graph, [BirthplaceCompleteness(threshold=0.9)])
```

- `name` identifies the metric in reports, thresholds, and comparisons.
- `category` groups the metric in reports. A metric of the graph alone is intrinsic.
- `unit` is the plural noun for what the metric counts.
- [`GraphView`][kgeval.GraphView] gives the graph's entities, facts, and chunks, as the
  [layout](configuration.md#graph-layout) reads them.
- `result` scores the share of units that pass, and records the metric as skipped when
  it evaluated none.

Give each finding a subject that names the unit at fault by content, as `graph.describe`
does. `compare` matches findings by subject, so a defect keeps its subject when a rebuild
assigns new element IDs.

## Add parameters and inputs

Declare parameters as fields, as `threshold` is. When a parameter changes the score,
override `signature` to include it, so that `compare` compares only scores measured
alike. A metric that reads an input, such as a graph schema, lists it in `needs` and
reads it with `self.require(inputs.schema, "schema")`.

## Ask a model

To decide units with a model, subclass [`SampledMetric`][kgeval.SampledMetric] instead.
Its `sample`, `ask_all`, and `estimate` methods pick a reproducible sample, ask the model
about each sampled unit, and estimate the score, with a confidence interval when the
sample leaves units out.
